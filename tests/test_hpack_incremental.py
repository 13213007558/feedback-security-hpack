"""
Tests for the incremental header block decoding interface:
``Decoder.begin_header_block`` / ``feed_data`` / ``end_header_block`` /
``abort_header_block``.
"""
import pytest

from hpack import (
    Decoder,
    Encoder,
    HPACKDecodingError,
    InvalidTableSizeError,
    OversizedHeaderListError,
)


def two_dependent_blocks():
    """
    Encodes two header blocks where the second one can only be decoded using
    the dynamic table entry created by the first one.
    """
    e = Encoder()
    first = e.encode([("custom-key", "custom-value")], huffman=False)
    second = e.encode(
        [("custom-key", "custom-value"), (":path", "/")], huffman=False,
    )
    return first, second


FIRST_HEADERS = [("custom-key", "custom-value")]
SECOND_HEADERS = [("custom-key", "custom-value"), (":path", "/")]


def feed_in_fragments(decoder, block, fragments):
    """
    Feeds ``block`` to ``decoder`` split at the given fragment boundaries.
    """
    decoder.begin_header_block()
    position = 0
    for size in fragments:
        decoder.feed_data(block[position:position + size])
        position += size
    decoder.feed_data(block[position:])
    return decoder.end_header_block()


class TestIncrementalDecoding:
    def test_byte_by_byte_two_dependent_blocks(self):
        """
        Two consecutive header blocks, the second depending on the dynamic
        table entry from the first, decode correctly when fed one byte at a
        time.
        """
        first, second = two_dependent_blocks()
        d = Decoder()

        d.begin_header_block()
        for i in range(len(first)):
            d.feed_data(first[i:i + 1])
        assert d.end_header_block() == FIRST_HEADERS

        d.begin_header_block()
        for i in range(len(second)):
            d.feed_data(second[i:i + 1])
        assert d.end_header_block() == SECOND_HEADERS

    def test_all_single_split_points(self):
        """
        Every possible two-fragment split of both blocks decodes to the same
        result as the one-shot decode.
        """
        first, second = two_dependent_blocks()

        for split in range(len(first) + 1):
            d = Decoder()
            assert feed_in_fragments(d, first, [split]) == FIRST_HEADERS

        for split in range(len(second) + 1):
            d = Decoder()
            d.decode(first)  # Prime the dynamic table.
            assert feed_in_fragments(d, second, [split]) == SECOND_HEADERS

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 8])
    def test_fixed_size_fragments_match_one_shot(self, chunk_size):
        """
        Fragmenting both blocks into fixed-size chunks yields the same
        headers as Decoder.decode.
        """
        first, second = two_dependent_blocks()
        d = Decoder()
        for block, expected in ((first, FIRST_HEADERS), (second, SECOND_HEADERS)):
            assert feed_in_fragments(
                d, block, [chunk_size] * len(block),
            ) == expected

    def test_dynamic_table_insertion_not_replayed(self):
        """
        Splitting a literal-with-indexing representation into fragments must
        not insert the entry into the dynamic table more than once.
        """
        first, _ = two_dependent_blocks()
        d = Decoder()
        d.begin_header_block()
        for i in range(len(first)):
            d.feed_data(first[i:i + 1])
            assert len(d.header_table.dynamic_entries) <= 1
        d.end_header_block()
        assert list(d.header_table.dynamic_entries) == [
            (b"custom-key", b"custom-value"),
        ]

    def test_string_length_varint_split_across_fragments(self):
        """
        A multi-byte string length varint may be split across fragments: the
        decoder waits for more data instead of failing.
        """
        e = Encoder()
        value = "y" * 300
        block = e.encode([("x-long", value)], huffman=False)

        # The value length (300) is encoded as the varint 0x7f 0xad 0x01
        # right after the literal name: split in the middle of it.
        prefix = b"\x40\x06x-long\x7f"
        assert block.startswith(prefix)

        d = Decoder()
        d.begin_header_block()
        d.feed_data(block[:len(prefix)])  # Ends inside the length varint.
        d.feed_data(block[len(prefix):])
        assert d.end_header_block() == [("x-long", value)]

    def test_index_varint_split_across_fragments(self):
        """
        A multi-byte index varint in an indexed representation may be split
        across fragments.
        """
        e = Encoder()
        headers = [("x-header-%d" % i, "v") for i in range(70)]
        setup = e.encode(headers, huffman=False)
        # The oldest dynamic entry is now at table index 131, so referencing
        # it needs a two-byte varint: 0xff 0x04.
        block = e.encode([("x-header-0", "v")], huffman=False)
        assert block == b"\xff\x04"

        d = Decoder()
        d.decode(setup)
        d.begin_header_block()
        d.feed_data(block[:1])  # Only the varint prefix byte.
        d.feed_data(block[1:])
        assert d.end_header_block() == [("x-header-0", "v")]

    def test_huffman_padding_split_across_fragments(self):
        """
        The EOS-prefix padding at the end of a Huffman string may arrive in a
        later fragment than the rest of the string.
        """
        e = Encoder()
        block = e.encode([("custom-key", "custom-header")], huffman=True)

        d = Decoder()
        d.begin_header_block()
        d.feed_data(block[:-1])  # Everything but the final padding byte.
        d.feed_data(block[-1:])
        assert d.end_header_block() == [("custom-key", "custom-header")]

    def test_huffman_string_split_at_every_byte(self):
        """
        Huffman-coded name and value decode correctly no matter where the
        fragment boundary falls inside them.
        """
        e = Encoder()
        block = e.encode([("custom-key", "custom-header")], huffman=True)

        for split in range(len(block) + 1):
            d = Decoder()
            assert feed_in_fragments(d, block, [split]) == [
                ("custom-key", "custom-header"),
            ]

    def test_invalid_huffman_padding_is_an_error(self):
        """
        A Huffman string whose padding is not a prefix of EOS is rejected as
        soon as the complete string has been received.
        """
        # Literal with indexing, literal name: huffman bit set, length 1,
        # byte 0x00. 0x00 decodes to the symbol '0' followed by 0-bits of
        # padding, which is not a valid EOS prefix.
        block = b"\x40\x81\x00\x00"

        d = Decoder()
        d.begin_header_block()
        with pytest.raises(HPACKDecodingError):
            d.feed_data(block)

    def test_table_size_update_split_across_fragments(self):
        """
        A table size update whose varint is split across fragments is applied
        once complete, before any header in the block.
        """
        e = Encoder()
        e.header_table_size = 128
        block = e.encode([("custom-key", "custom-value")], huffman=False)
        assert block[0] & 0x20  # Starts with a table size update.

        d = Decoder()
        d.begin_header_block()
        d.feed_data(block[:1])  # Update varint may be incomplete here.
        d.feed_data(block[1:])
        assert d.end_header_block() == FIRST_HEADERS
        assert d.header_table_size == 128

    def test_table_size_update_too_large_mid_block(self):
        """
        A table size update exceeding the maximum is rejected as soon as its
        varint completes, even mid-stream.
        """
        d = Decoder()
        d.max_allowed_table_size = 100

        d.begin_header_block()
        d.feed_data(b"\x3f")  # Update prefix: needs more bytes, not an error.
        with pytest.raises(InvalidTableSizeError):
            d.feed_data(b"\x46")  # Completes the update to 101 > 100.

    def test_table_size_update_after_header_is_an_error(self):
        """
        A table size update appearing after a header field is rejected, even
        when it arrives in a later fragment than the header.
        """
        block = b"\x40\x01a\x01b" + b"\x20"  # Literal header, then update.

        d = Decoder()
        d.begin_header_block()
        d.feed_data(block[:5])  # The complete header field.
        with pytest.raises(HPACKDecodingError, match="Table size update"):
            d.feed_data(block[5:])

    def test_end_with_incomplete_block_is_truncation(self):
        """
        Declaring the block finished while a representation is incomplete is
        a truncation error, not "need more data".
        """
        first, _ = two_dependent_blocks()
        d = Decoder()
        d.begin_header_block()
        d.feed_data(first[:-1])
        with pytest.raises(HPACKDecodingError, match="Truncated"):
            d.end_header_block()

    def test_decoder_unusable_after_error(self):
        """
        After a decoding error the decoder refuses to decode further header
        blocks: the caller must build a new decoding context.
        """
        d = Decoder()
        d.begin_header_block()
        with pytest.raises(HPACKDecodingError):
            d.feed_data(b"\xff\xff\xff\xff\xff\xff")  # Over-long varint.

        with pytest.raises(HPACKDecodingError):
            d.begin_header_block()
        with pytest.raises(HPACKDecodingError):
            d.feed_data(b"\x82")
        with pytest.raises(HPACKDecodingError):
            d.end_header_block()
        with pytest.raises(HPACKDecodingError):
            d.decode(b"\x82")

        # A fresh decoder is unaffected.
        assert Decoder().decode(b"\x82") == [(":method", "GET")]

    def test_decoder_unusable_after_truncated_block(self):
        """
        A truncation error at end_header_block also permanently fails the
        decoder.
        """
        first, _ = two_dependent_blocks()
        d = Decoder()
        d.begin_header_block()
        d.feed_data(first[:-1])
        with pytest.raises(HPACKDecodingError):
            d.end_header_block()
        with pytest.raises(HPACKDecodingError):
            d.begin_header_block()

    def test_abort_invalidates_decoder(self):
        """
        Explicitly aborting a header block permanently fails the decoder.
        """
        d = Decoder()
        d.begin_header_block()
        d.feed_data(b"\x40")
        d.abort_header_block()

        with pytest.raises(HPACKDecodingError):
            d.begin_header_block()
        with pytest.raises(HPACKDecodingError):
            d.decode(b"\x82")

    def test_max_header_list_size_enforced_across_fragments(self):
        """
        The header list size limit applies to the whole block and is
        enforced as soon as a fragment pushes it over the limit.
        """
        block = b"\x40\x01a\x01b\x40\x01c\x01d"  # Two literal headers.
        d = Decoder(max_header_list_size=64)  # Each entry inflates by 34.

        d.begin_header_block()
        with pytest.raises(OversizedHeaderListError):
            for i in range(len(block)):
                d.feed_data(block[i:i + 1])

    def test_api_misuse_does_not_poison_decoder(self):
        """
        Calling the incremental methods out of order is an error, but does
        not corrupt the compression state.
        """
        d = Decoder()
        with pytest.raises(HPACKDecodingError):
            d.feed_data(b"\x82")
        with pytest.raises(HPACKDecodingError):
            d.end_header_block()

        d.begin_header_block()
        with pytest.raises(HPACKDecodingError):
            d.begin_header_block()
        d.feed_data(b"\x82")
        assert d.end_header_block() == [(":method", "GET")]

    def test_decode_matches_incremental_result(self):
        """
        The one-shot decode and the incremental interface share semantics:
        both accept the same blocks and deliver the same headers.
        """
        first, second = two_dependent_blocks()

        one_shot = Decoder()
        incremental = Decoder()
        for block in (first, second):
            expected = one_shot.decode(block)
            incremental.begin_header_block()
            incremental.feed_data(block)
            assert incremental.end_header_block() == expected

    def test_empty_block(self):
        """
        An empty header block decodes to an empty header list.
        """
        d = Decoder()
        d.begin_header_block()
        assert d.end_header_block() == []
