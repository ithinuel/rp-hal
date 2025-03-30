/// Register `UARTDMACR` reader
pub type R = crate::register_blocks::R<UARTDMACR_SPEC>;
/// Register `UARTDMACR` writer
pub type W = crate::register_blocks::W<UARTDMACR_SPEC>;
/// Field `RXDMAE` reader - Receive DMA enable. If this bit is set to 1, DMA for the receive FIFO is enabled.
pub type RXDMAE_R = crate::register_blocks::BitReader;
/// Field `RXDMAE` writer - Receive DMA enable. If this bit is set to 1, DMA for the receive FIFO is enabled.
pub type RXDMAE_W<'a, REG> = crate::register_blocks::BitWriter<'a, REG>;
/// Field `TXDMAE` reader - Transmit DMA enable. If this bit is set to 1, DMA for the transmit FIFO is enabled.
pub type TXDMAE_R = crate::register_blocks::BitReader;
/// Field `TXDMAE` writer - Transmit DMA enable. If this bit is set to 1, DMA for the transmit FIFO is enabled.
pub type TXDMAE_W<'a, REG> = crate::register_blocks::BitWriter<'a, REG>;
/// Field `DMAONERR` reader - DMA on error. If this bit is set to 1, the DMA receive request outputs, UARTRXDMASREQ or UARTRXDMABREQ, are disabled when the UART error interrupt is asserted.
pub type DMAONERR_R = crate::register_blocks::BitReader;
/// Field `DMAONERR` writer - DMA on error. If this bit is set to 1, the DMA receive request outputs, UARTRXDMASREQ or UARTRXDMABREQ, are disabled when the UART error interrupt is asserted.
pub type DMAONERR_W<'a, REG> = crate::register_blocks::BitWriter<'a, REG>;
impl R {
    /// Bit 0 - Receive DMA enable. If this bit is set to 1, DMA for the receive FIFO is enabled.
    #[inline(always)]
    pub fn rxdmae(&self) -> RXDMAE_R {
        RXDMAE_R::new((self.bits & 1) != 0)
    }
    /// Bit 1 - Transmit DMA enable. If this bit is set to 1, DMA for the transmit FIFO is enabled.
    #[inline(always)]
    pub fn txdmae(&self) -> TXDMAE_R {
        TXDMAE_R::new(((self.bits >> 1) & 1) != 0)
    }
    /// Bit 2 - DMA on error. If this bit is set to 1, the DMA receive request outputs, UARTRXDMASREQ or UARTRXDMABREQ, are disabled when the UART error interrupt is asserted.
    #[inline(always)]
    pub fn dmaonerr(&self) -> DMAONERR_R {
        DMAONERR_R::new(((self.bits >> 2) & 1) != 0)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTDMACR")
            .field("dmaonerr", &self.dmaonerr())
            .field("txdmae", &self.txdmae())
            .field("rxdmae", &self.rxdmae())
            .finish()
    }
}
impl W {
    /// Bit 0 - Receive DMA enable. If this bit is set to 1, DMA for the receive FIFO is enabled.
    #[inline(always)]
    pub fn rxdmae(&mut self) -> RXDMAE_W<UARTDMACR_SPEC> {
        RXDMAE_W::new(self, 0)
    }
    /// Bit 1 - Transmit DMA enable. If this bit is set to 1, DMA for the transmit FIFO is enabled.
    #[inline(always)]
    pub fn txdmae(&mut self) -> TXDMAE_W<UARTDMACR_SPEC> {
        TXDMAE_W::new(self, 1)
    }
    /// Bit 2 - DMA on error. If this bit is set to 1, the DMA receive request outputs, UARTRXDMASREQ or UARTRXDMABREQ, are disabled when the UART error interrupt is asserted.
    #[inline(always)]
    pub fn dmaonerr(&mut self) -> DMAONERR_W<UARTDMACR_SPEC> {
        DMAONERR_W::new(self, 2)
    }
}
/// DMA Control Register, UARTDMACR  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartdmacr::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartdmacr::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTDMACR_SPEC;
impl crate::register_blocks::RegisterSpec for UARTDMACR_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartdmacr::R`](R) reader structure
impl crate::register_blocks::Readable for UARTDMACR_SPEC {}
/// `write(|w| ..)` method takes [`uartdmacr::W`](W) writer structure
impl crate::register_blocks::Writable for UARTDMACR_SPEC {
    type Safety = crate::register_blocks::Unsafe;
}
/// `reset()` method sets UARTDMACR to value 0
impl crate::register_blocks::Resettable for UARTDMACR_SPEC {}
