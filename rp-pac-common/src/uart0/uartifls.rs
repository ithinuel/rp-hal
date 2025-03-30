/// Register `UARTIFLS` reader
pub type R = crate::register_blocks::R<UARTIFLS_SPEC>;
/// Register `UARTIFLS` writer
pub type W = crate::register_blocks::W<UARTIFLS_SPEC>;
/// Field `TXIFLSEL` reader - Transmit interrupt FIFO level select. The trigger points for the transmit interrupt are as follows: b000 = Transmit FIFO becomes <= 1 / 8 full b001 = Transmit FIFO becomes <= 1 / 4 full b010 = Transmit FIFO becomes <= 1 / 2 full b011 = Transmit FIFO becomes <= 3 / 4 full b100 = Transmit FIFO becomes <= 7 / 8 full b101-b111 = reserved.
pub type TXIFLSEL_R = crate::register_blocks::FieldReader;
/// Field `TXIFLSEL` writer - Transmit interrupt FIFO level select. The trigger points for the transmit interrupt are as follows: b000 = Transmit FIFO becomes <= 1 / 8 full b001 = Transmit FIFO becomes <= 1 / 4 full b010 = Transmit FIFO becomes <= 1 / 2 full b011 = Transmit FIFO becomes <= 3 / 4 full b100 = Transmit FIFO becomes <= 7 / 8 full b101-b111 = reserved.
pub type TXIFLSEL_W<'a, REG> = crate::register_blocks::FieldWriter<'a, REG, 3>;
/// Field `RXIFLSEL` reader - Receive interrupt FIFO level select. The trigger points for the receive interrupt are as follows: b000 = Receive FIFO becomes >= 1 / 8 full b001 = Receive FIFO becomes >= 1 / 4 full b010 = Receive FIFO becomes >= 1 / 2 full b011 = Receive FIFO becomes >= 3 / 4 full b100 = Receive FIFO becomes >= 7 / 8 full b101-b111 = reserved.
pub type RXIFLSEL_R = crate::register_blocks::FieldReader;
/// Field `RXIFLSEL` writer - Receive interrupt FIFO level select. The trigger points for the receive interrupt are as follows: b000 = Receive FIFO becomes >= 1 / 8 full b001 = Receive FIFO becomes >= 1 / 4 full b010 = Receive FIFO becomes >= 1 / 2 full b011 = Receive FIFO becomes >= 3 / 4 full b100 = Receive FIFO becomes >= 7 / 8 full b101-b111 = reserved.
pub type RXIFLSEL_W<'a, REG> = crate::register_blocks::FieldWriter<'a, REG, 3>;
impl R {
    /// Bits 0:2 - Transmit interrupt FIFO level select. The trigger points for the transmit interrupt are as follows: b000 = Transmit FIFO becomes <= 1 / 8 full b001 = Transmit FIFO becomes <= 1 / 4 full b010 = Transmit FIFO becomes <= 1 / 2 full b011 = Transmit FIFO becomes <= 3 / 4 full b100 = Transmit FIFO becomes <= 7 / 8 full b101-b111 = reserved.
    #[inline(always)]
    pub fn txiflsel(&self) -> TXIFLSEL_R {
        TXIFLSEL_R::new((self.bits & 7) as u8)
    }
    /// Bits 3:5 - Receive interrupt FIFO level select. The trigger points for the receive interrupt are as follows: b000 = Receive FIFO becomes >= 1 / 8 full b001 = Receive FIFO becomes >= 1 / 4 full b010 = Receive FIFO becomes >= 1 / 2 full b011 = Receive FIFO becomes >= 3 / 4 full b100 = Receive FIFO becomes >= 7 / 8 full b101-b111 = reserved.
    #[inline(always)]
    pub fn rxiflsel(&self) -> RXIFLSEL_R {
        RXIFLSEL_R::new(((self.bits >> 3) & 7) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTIFLS")
            .field("rxiflsel", &self.rxiflsel())
            .field("txiflsel", &self.txiflsel())
            .finish()
    }
}
impl W {
    /// Bits 0:2 - Transmit interrupt FIFO level select. The trigger points for the transmit interrupt are as follows: b000 = Transmit FIFO becomes <= 1 / 8 full b001 = Transmit FIFO becomes <= 1 / 4 full b010 = Transmit FIFO becomes <= 1 / 2 full b011 = Transmit FIFO becomes <= 3 / 4 full b100 = Transmit FIFO becomes <= 7 / 8 full b101-b111 = reserved.
    #[inline(always)]
    pub fn txiflsel(&mut self) -> TXIFLSEL_W<UARTIFLS_SPEC> {
        TXIFLSEL_W::new(self, 0)
    }
    /// Bits 3:5 - Receive interrupt FIFO level select. The trigger points for the receive interrupt are as follows: b000 = Receive FIFO becomes >= 1 / 8 full b001 = Receive FIFO becomes >= 1 / 4 full b010 = Receive FIFO becomes >= 1 / 2 full b011 = Receive FIFO becomes >= 3 / 4 full b100 = Receive FIFO becomes >= 7 / 8 full b101-b111 = reserved.
    #[inline(always)]
    pub fn rxiflsel(&mut self) -> RXIFLSEL_W<UARTIFLS_SPEC> {
        RXIFLSEL_W::new(self, 3)
    }
}
/// Interrupt FIFO Level Select Register, UARTIFLS  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartifls::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartifls::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTIFLS_SPEC;
impl crate::register_blocks::RegisterSpec for UARTIFLS_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartifls::R`](R) reader structure
impl crate::register_blocks::Readable for UARTIFLS_SPEC {}
/// `write(|w| ..)` method takes [`uartifls::W`](W) writer structure
impl crate::register_blocks::Writable for UARTIFLS_SPEC {
    type Safety = crate::register_blocks::Unsafe;
}
/// `reset()` method sets UARTIFLS to value 0x12
impl crate::register_blocks::Resettable for UARTIFLS_SPEC {
    const RESET_VALUE: u32 = 0x12;
}
