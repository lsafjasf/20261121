pub struct BitWriter {
    bytes: Vec<u8>,
    bit: u8,
}

impl BitWriter {
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            bit: 0,
        }
    }

    pub fn write_u64(&mut self, value: u64, count: u32) {
        debug_assert!(count <= 64);
        for shift in (0..count).rev() {
            let bit = ((value >> shift) & 1) as u8;
            if self.bit == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.last_mut().expect("a byte was just created");
            *last |= bit << (7 - self.bit);
            self.bit = (self.bit + 1) & 7;
        }
    }

    pub fn write_signed(&mut self, value: i64, count: u32) {
        let mask = if count == 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        self.write_u64((value as u64) & mask, count);
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bit, 0);
        self.bytes.extend_from_slice(bytes);
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.bit = 0;
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_packs_signed_values_msb_first() {
        let mut writer = BitWriter::with_capacity(4);
        writer.write_signed(-3, 5);
        writer.write_signed(17, 9);
        writer.write_u64(0b101, 3);
        let bytes = writer.finish();
        assert_eq!(bytes, [0b1110_1000, 0b0100_0110, 0b1000_0000]);
    }
}
