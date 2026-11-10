"""
Tests for the Encoder's failure contract.

A single ``encode`` call is transactional: if the caller's header generator,
the value-to-bytes conversion, or input validation raises before a complete
block is produced, none of the bytes describing that block have been emitted.
The encoder therefore rolls the dynamic table, the configured table size and
the pending table size updates back to the state they were in before the call,
so the same Encoder can be retried without rebuilding the compression context
and stay in lock-step with a Decoder that never received the failed block.
"""
import pytest

from hpack import Decoder, Encoder, HeaderTuple, NeverIndexedHeaderTuple


class ExplodingValue:
    """A value whose conversion to bytes deliberately fails."""

    def __str__(self) -> str:
        raise RuntimeError("cannot convert value to bytes")


def failing_headers(fail_at, exc_type=RuntimeError):
    """
    Yields three headers, raising ``exc_type`` while producing the header at
    the zero-based position ``fail_at``. Headers produced before the failure
    already mutate the dynamic table.
    """
    for position in range(3):

        def raise_later(exc_type=exc_type) -> None:
            raise exc_type("generator failure")

        if position == fail_at:
            raise_later()
        yield (f"key{position}", f"value{position}")


class TestEncoderFailureRollback:
    @pytest.mark.parametrize("fail_at", [0, 1, 2])
    def test_generator_failure_rolls_back_dynamic_table(self, fail_at):
        encoder = Encoder()
        synced = [(b"seed", b"x")]
        encoder.encode([(b"seed", b"x")])
        expected = list(encoder.header_table.dynamic_entries)
        assert expected == synced

        with pytest.raises(RuntimeError):
            encoder.encode(failing_headers(fail_at))

        # The failed call left no trace on the dynamic table.
        assert list(encoder.header_table.dynamic_entries) == expected

    @pytest.mark.parametrize("fail_at", [0, 1, 2])
    def test_invalid_input_rolls_back_dynamic_table(self, fail_at):
        encoder = Encoder()
        encoder.encode([(b"seed", b"x")])
        expected = list(encoder.header_table.dynamic_entries)

        headers = []
        for position in range(3):
            if position == fail_at:
                headers.append(("key", ExplodingValue()))
            else:
                headers.append((f"key{position}", f"value{position}"))

        with pytest.raises(RuntimeError):
            encoder.encode(headers)

        assert list(encoder.header_table.dynamic_entries) == expected

    @pytest.mark.parametrize("fail_at", [0, 1, 2])
    def test_malformed_header_rolls_back_dynamic_table(self, fail_at):
        encoder = Encoder()
        encoder.encode([(b"seed", b"x")])
        expected = list(encoder.header_table.dynamic_entries)

        headers = []
        for position in range(3):
            if position == fail_at:
                headers.append(("one-tuple",))
            else:
                headers.append((f"key{position}", f"value{position}"))

        with pytest.raises(IndexError):
            encoder.encode(headers)

        assert list(encoder.header_table.dynamic_entries) == expected

    def test_failure_restores_entries_evicted_during_the_failed_encode(self):
        encoder = Encoder()

        # A small table that holds exactly two 40-byte entries.
        encoder.header_table_size = 128
        encoder.encode([("aaaa", "bbbb"), ("cccc", "dddd")])
        at_encode_entry = encoder.header_table.snapshot()
        assert len(at_encode_entry[3]) == 2

        def failing():
            # This 63-byte entry forces an eviction of a prior entry before
            # the generator raises on the next header.
            yield ("long-enough-name", "long-enough-value")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            encoder.encode(failing())

        # The eviction and the insert made by the aborted call are undone.
        assert encoder.header_table_size == at_encode_entry[0]
        assert encoder.header_table.resized == at_encode_entry[2]
        assert list(encoder.header_table.dynamic_entries) == \
            list(at_encode_entry[3])
        assert len(encoder.header_table.dynamic_entries) == 2

    def test_external_resize_persists_but_encode_mutations_roll_back(self):
        encoder = Encoder()
        encoder.encode([("aaaa", "bbbb"), ("cccc", "dddd")])

        # The shrink is requested outside encode and must persist, even though
        # it evicts an entry.
        encoder.header_table_size = 64
        at_encode_entry = encoder.header_table.snapshot()
        assert len(at_encode_entry[3]) == 1

        with pytest.raises(RuntimeError):
            encoder.encode(failing_headers(fail_at=1))

        # The externally requested size survives; only in-call mutations undo.
        assert encoder.header_table_size == 64
        assert list(encoder.header_table.dynamic_entries) == \
            list(at_encode_entry[3])

    def test_pending_table_size_changes_survive_failure(self):
        encoder = Encoder()

        encoder.header_table_size = 40
        encoder.header_table_size = 100
        pending = encoder.table_size_changes.copy()

        with pytest.raises(RuntimeError):
            encoder.encode(failing_headers(fail_at=2))

        # The updates were not consumed by the aborted call and stay pending.
        assert encoder.table_size_changes == pending
        assert encoder.header_table.resized is True

    def test_successful_encode_commits_and_clears_pending_changes(self):
        encoder = Encoder()
        encoder.header_table_size = 40
        encoder.header_table_size = 100

        block = encoder.encode([(":method", "GET")])

        assert encoder.table_size_changes == []
        assert encoder.header_table.resized is False
        assert Decoder().decode(block) == [(":method", "GET")]

    def test_sensitive_headers_are_not_rolled_back_into_table(self):
        encoder = Encoder()
        encoder.encode([(b"seed", b"x")])
        expected = list(encoder.header_table.dynamic_entries)

        headers = [
            NeverIndexedHeaderTuple("secret", "shh"),
            ("key1", "value1"),
        ]

        def failing():
            yield headers[0]
            yield headers[1]
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            encoder.encode(failing())

        assert list(encoder.header_table.dynamic_entries) == expected

    def test_rollback_is_reusable_across_repeated_failures(self):
        encoder = Encoder()
        encoder.encode([(b"seed", b"x")])
        expected = list(encoder.header_table.dynamic_entries)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                encoder.encode(failing_headers(fail_at=1))
            assert list(encoder.header_table.dynamic_entries) == expected

        block = encoder.encode([("key0", "value0")])
        assert Decoder().decode(block) == [("key0", "value0")]


class TestEncoderFailureRoundTrip:
    def test_shrink_grow_failure_retry_sequence_decodes_on_fresh_decoder(self):
        """
        Drives one Encoder through the required short sequence: a successful
        block, a shrink, a grow, a failure on the second header, then a retry.
        A single Decoder only ever receives the blocks that were returned
        successfully, and after each one the two compression contexts hold the
        same dynamic entries and agree on the decoded headers.
        """
        encoder = Encoder()
        decoder = Decoder()

        first = [
            (":method", "GET"),
            ("first-name", "first-value"),
            ("second-name", "second-value"),
        ]
        first_block = encoder.encode(first, huffman=False)
        assert decoder.decode(first_block) == first
        assert list(encoder.header_table.dynamic_entries) == \
            list(decoder.header_table.dynamic_entries)

        # Shrink then grow: both become pending table size updates. The shrink
        # evicts an entry; because these resizes are requested by the caller
        # before encode, they persist even when the next encode aborts.
        encoder.header_table_size = 64
        encoder.header_table_size = 256
        assert encoder.header_table_size == 256
        pending = encoder.table_size_changes.copy()
        at_encode_entry = encoder.header_table.snapshot()

        # Failure on the second header: the first header already touched the
        # dynamic table before the generator raised.
        def failing_on_second():
            yield ("failed-first", "value")
            raise RuntimeError("boom on second header")

        with pytest.raises(RuntimeError):
            encoder.encode(failing_on_second(), huffman=False)

        # Everything the aborted call did is undone, while the externally
        # requested size and still-pending updates are preserved.
        assert list(encoder.header_table.dynamic_entries) == \
            list(at_encode_entry[3])
        assert encoder.header_table_size == at_encode_entry[0]
        assert encoder.table_size_changes == pending
        assert encoder.header_table.resized is True

        # Retry with valid input. This block carries the pending table size
        # updates and must decode on the Decoder, which never saw failed data.
        retry = [
            (":method", "GET"),
            ("retry-name", "retry-value"),
        ]
        retry_block = encoder.encode(retry, huffman=False)
        assert decoder.decode(retry_block) == retry

        # Both ends now agree on table size and on every dynamic entry.
        assert encoder.header_table_size == decoder.header_table_size == 256
        assert list(encoder.header_table.dynamic_entries) == \
            list(decoder.header_table.dynamic_entries)
        assert encoder.table_size_changes == []

    def test_retry_block_is_self_contained_for_unaware_decoder(self):
        """
        A fresh Decoder that only receives successful blocks (including the
        retry after a failure) can decode every block, because the failed
        block never poisoned the Encoder's table.
        """
        encoder = Encoder()
        decoder = Decoder()

        decoder.decode(encoder.encode([("seed", "x")]))

        with pytest.raises(RuntimeError):
            encoder.encode(failing_headers(fail_at=2))

        # A header that was inserted only during the failed call is retried and
        # must be sent as a fresh literal, not an index the decoder lacks.
        retry_block = encoder.encode([("key0", "value0")])
        assert decoder.decode(retry_block) == [("key0", "value0")]
        assert list(encoder.header_table.dynamic_entries) == \
            list(decoder.header_table.dynamic_entries)

    def test_header_tuples_survive_failure_and_retry(self):
        encoder = Encoder()
        decoder = Decoder()

        decoder.decode(encoder.encode([HeaderTuple("seed", "x")]))
        before = list(encoder.header_table.dynamic_entries)

        def failing():
            yield HeaderTuple("a", "1")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            encoder.encode(failing())

        assert list(encoder.header_table.dynamic_entries) == before
        block = encoder.encode([HeaderTuple("a", "1")])
        decoded = decoder.decode(block)
        assert decoded == [("a", "1")]
        assert all(isinstance(header, HeaderTuple) for header in decoded)
