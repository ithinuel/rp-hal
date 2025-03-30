/// Register `UARTDR` reader
pub type R = crate::register_blocks::R<UARTDR_SPEC>;
/// Register `UARTDR` writer
pub type W = crate::register_blocks::W<UARTDR_SPEC>;
/// Field `DATA` reader - Receive (read) data character. Transmit (write) data character.  /// /// <div class=\"warning\">The field is <b>modified</b> in some way after a read operation.</div>
pub type DATA_R = crate::register_blocks::FieldReader;
/// Field `DATA` writer - Receive (read) data character. Transmit (write) data character.
pub type DATA_W<'a, REG> = crate::register_blocks::FieldWriter<'a, REG, 8>;
/// Field `FE` reader - Framing error. When set to 1, it indicates that the received character did not have a valid stop bit (a valid stop bit is 1). In FIFO mode, this error is associated with the character at the top of the FIFO.
pub type FE_R = crate::register_blocks::BitReader;
/// Field `PE` reader - Parity error. When set to 1, it indicates that the parity of the received data character does not match the parity that the EPS and SPS bits in the Line Control Register, UARTLCR_H. In FIFO mode, this error is associated with the character at the top of the FIFO.
pub type PE_R = crate::register_blocks::BitReader;
/// Field `BE` reader - Break error. This bit is set to 1 if a break condition was detected, indicating that the received data input was held LOW for longer than a full-word transmission time (defined as start, data, parity and stop bits). In FIFO mode, this error is associated with the character at the top of the FIFO. When a break occurs, only one 0 character is loaded into the FIFO. The next character is only enabled after the receive data input goes to a 1 (marking state), and the next valid start bit is received.
pub type BE_R = crate::register_blocks::BitReader;
/// Field `OE` reader - Overrun error. This bit is set to 1 if data is received and the receive FIFO is already full. This is cleared to 0 once there is an empty space in the FIFO and a new character can be written to it.
pub type OE_R = crate::register_blocks::BitReader;
impl R {
    /// Bits 0:7 - Receive (read) data character. Transmit (write) data character.
    #[inline(always)]
    pub fn data(&self) -> DATA_R {
        DATA_R::new((self.bits & 0xff) as u8)
    }
    /// Bit 8 - Framing error. When set to 1, it indicates that the received character did not have a valid stop bit (a valid stop bit is 1). In FIFO mode, this error is associated with the character at the top of the FIFO.
    #[inline(always)]
    pub fn fe(&self) -> FE_R {
        FE_R::new(((self.bits >> 8) & 1) != 0)
    }
    /// Bit 9 - Parity error. When set to 1, it indicates that the parity of the received data character does not match the parity that the EPS and SPS bits in the Line Control Register, UARTLCR_H. In FIFO mode, this error is associated with the character at the top of the FIFO.
    #[inline(always)]
    pub fn pe(&self) -> PE_R {
        PE_R::new(((self.bits >> 9) & 1) != 0)
    }
    /// Bit 10 - Break error. This bit is set to 1 if a break condition was detected, indicating that the received data input was held LOW for longer than a full-word transmission time (defined as start, data, parity and stop bits). In FIFO mode, this error is associated with the character at the top of the FIFO. When a break occurs, only one 0 character is loaded into the FIFO. The next character is only enabled after the receive data input goes to a 1 (marking state), and the next valid start bit is received.
    #[inline(always)]
    pub fn be(&self) -> BE_R {
        BE_R::new(((self.bits >> 10) & 1) != 0)
    }
    /// Bit 11 - Overrun error. This bit is set to 1 if data is received and the receive FIFO is already full. This is cleared to 0 once there is an empty space in the FIFO and a new character can be written to it.
    #[inline(always)]
    pub fn oe(&self) -> OE_R {
        OE_R::new(((self.bits >> 11) & 1) != 0)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTDR")
            .field("oe", &self.oe())
            .field("be", &self.be())
            .field("pe", &self.pe())
            .field("fe", &self.fe())
            .finish()
    }
}
impl W {
    /// Bits 0:7 - Receive (read) data character. Transmit (write) data character.
    #[inline(always)]
    pub fn data(&mut self) -> DATA_W<UARTDR_SPEC> {
        DATA_W::new(self, 0)
    }
}
/// Data Register, UARTDR  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartdr::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartdr::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTDR_SPEC;
impl crate::register_blocks::RegisterSpec for UARTDR_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartdr::R`](R) reader structure
impl crate::register_blocks::Readable for UARTDR_SPEC {}
/// `write(|w| ..)` method takes [`uartdr::W`](W) writer structure
impl crate::register_blocks::Writable for UARTDR_SPEC {
    type Safety = crate::register_blocks::Unsafe;
}
/// `reset()` method sets UARTDR to value 0
impl crate::register_blocks::Resettable for UARTDR_SPEC {}
