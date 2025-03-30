/// Register `UARTRSR` reader
pub type R = crate::register_blocks::R<UARTRSR_SPEC>;
/// Register `UARTRSR` writer
pub type W = crate::register_blocks::W<UARTRSR_SPEC>;
/// Field `FE` reader - Framing error. When set to 1, it indicates that the received character did not have a valid stop bit (a valid stop bit is 1). This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
pub type FE_R = crate::register_blocks::BitReader;
/// Field `FE` writer - Framing error. When set to 1, it indicates that the received character did not have a valid stop bit (a valid stop bit is 1). This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
pub type FE_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `PE` reader - Parity error. When set to 1, it indicates that the parity of the received data character does not match the parity that the EPS and SPS bits in the Line Control Register, UARTLCR_H. This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
pub type PE_R = crate::register_blocks::BitReader;
/// Field `PE` writer - Parity error. When set to 1, it indicates that the parity of the received data character does not match the parity that the EPS and SPS bits in the Line Control Register, UARTLCR_H. This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
pub type PE_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `BE` reader - Break error. This bit is set to 1 if a break condition was detected, indicating that the received data input was held LOW for longer than a full-word transmission time (defined as start, data, parity, and stop bits). This bit is cleared to 0 after a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO. When a break occurs, only one 0 character is loaded into the FIFO. The next character is only enabled after the receive data input goes to a 1 (marking state) and the next valid start bit is received.
pub type BE_R = crate::register_blocks::BitReader;
/// Field `BE` writer - Break error. This bit is set to 1 if a break condition was detected, indicating that the received data input was held LOW for longer than a full-word transmission time (defined as start, data, parity, and stop bits). This bit is cleared to 0 after a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO. When a break occurs, only one 0 character is loaded into the FIFO. The next character is only enabled after the receive data input goes to a 1 (marking state) and the next valid start bit is received.
pub type BE_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `OE` reader - Overrun error. This bit is set to 1 if data is received and the FIFO is already full. This bit is cleared to 0 by a write to UARTECR. The FIFO contents remain valid because no more data is written when the FIFO is full, only the contents of the shift register are overwritten. The CPU must now read the data, to empty the FIFO.
pub type OE_R = crate::register_blocks::BitReader;
/// Field `OE` writer - Overrun error. This bit is set to 1 if data is received and the FIFO is already full. This bit is cleared to 0 by a write to UARTECR. The FIFO contents remain valid because no more data is written when the FIFO is full, only the contents of the shift register are overwritten. The CPU must now read the data, to empty the FIFO.
pub type OE_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
impl R {
    /// Bit 0 - Framing error. When set to 1, it indicates that the received character did not have a valid stop bit (a valid stop bit is 1). This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
    #[inline(always)]
    pub fn fe(&self) -> FE_R {
        FE_R::new((self.bits & 1) != 0)
    }
    /// Bit 1 - Parity error. When set to 1, it indicates that the parity of the received data character does not match the parity that the EPS and SPS bits in the Line Control Register, UARTLCR_H. This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
    #[inline(always)]
    pub fn pe(&self) -> PE_R {
        PE_R::new(((self.bits >> 1) & 1) != 0)
    }
    /// Bit 2 - Break error. This bit is set to 1 if a break condition was detected, indicating that the received data input was held LOW for longer than a full-word transmission time (defined as start, data, parity, and stop bits). This bit is cleared to 0 after a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO. When a break occurs, only one 0 character is loaded into the FIFO. The next character is only enabled after the receive data input goes to a 1 (marking state) and the next valid start bit is received.
    #[inline(always)]
    pub fn be(&self) -> BE_R {
        BE_R::new(((self.bits >> 2) & 1) != 0)
    }
    /// Bit 3 - Overrun error. This bit is set to 1 if data is received and the FIFO is already full. This bit is cleared to 0 by a write to UARTECR. The FIFO contents remain valid because no more data is written when the FIFO is full, only the contents of the shift register are overwritten. The CPU must now read the data, to empty the FIFO.
    #[inline(always)]
    pub fn oe(&self) -> OE_R {
        OE_R::new(((self.bits >> 3) & 1) != 0)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTRSR")
            .field("oe", &self.oe())
            .field("be", &self.be())
            .field("pe", &self.pe())
            .field("fe", &self.fe())
            .finish()
    }
}
impl W {
    /// Bit 0 - Framing error. When set to 1, it indicates that the received character did not have a valid stop bit (a valid stop bit is 1). This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
    #[inline(always)]
    pub fn fe(&mut self) -> FE_W<UARTRSR_SPEC> {
        FE_W::new(self, 0)
    }
    /// Bit 1 - Parity error. When set to 1, it indicates that the parity of the received data character does not match the parity that the EPS and SPS bits in the Line Control Register, UARTLCR_H. This bit is cleared to 0 by a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO.
    #[inline(always)]
    pub fn pe(&mut self) -> PE_W<UARTRSR_SPEC> {
        PE_W::new(self, 1)
    }
    /// Bit 2 - Break error. This bit is set to 1 if a break condition was detected, indicating that the received data input was held LOW for longer than a full-word transmission time (defined as start, data, parity, and stop bits). This bit is cleared to 0 after a write to UARTECR. In FIFO mode, this error is associated with the character at the top of the FIFO. When a break occurs, only one 0 character is loaded into the FIFO. The next character is only enabled after the receive data input goes to a 1 (marking state) and the next valid start bit is received.
    #[inline(always)]
    pub fn be(&mut self) -> BE_W<UARTRSR_SPEC> {
        BE_W::new(self, 2)
    }
    /// Bit 3 - Overrun error. This bit is set to 1 if data is received and the FIFO is already full. This bit is cleared to 0 by a write to UARTECR. The FIFO contents remain valid because no more data is written when the FIFO is full, only the contents of the shift register are overwritten. The CPU must now read the data, to empty the FIFO.
    #[inline(always)]
    pub fn oe(&mut self) -> OE_W<UARTRSR_SPEC> {
        OE_W::new(self, 3)
    }
}
/// Receive Status Register/Error Clear Register, UARTRSR/UARTECR  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartrsr::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartrsr::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTRSR_SPEC;
impl crate::register_blocks::RegisterSpec for UARTRSR_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartrsr::R`](R) reader structure
impl crate::register_blocks::Readable for UARTRSR_SPEC {}
/// `write(|w| ..)` method takes [`uartrsr::W`](W) writer structure
impl crate::register_blocks::Writable for UARTRSR_SPEC {
    type Safety = crate::register_blocks::Unsafe;
    const ONE_TO_MODIFY_FIELDS_BITMAP: u32 = 0x0f;
}
/// `reset()` method sets UARTRSR to value 0
impl crate::register_blocks::Resettable for UARTRSR_SPEC {}
