use training::quantised_len;

#[test]
fn matches_the_sizes_mudskipper_expects() {
    // (768·N + N + 2N + 1) i16 values, padded to 64 bytes, worked out by hand.
    assert_eq!(quantised_len(128), 197_440);
    assert_eq!(quantised_len(256), 394_816);
}
