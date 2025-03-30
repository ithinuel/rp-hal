/// Register `UARTRIS` reader
pub type R = crate::register_blocks::R<UARTRIS_SPEC>;
/// Field `RIRMIS` reader - nUARTRI modem interrupt status. Returns the raw interrupt state of the UARTRIINTR interrupt.
pub type RIRMIS_R = crate::register_blocks::BitReader;
/// Field `CTSRMIS` reader - nUARTCTS modem interrupt status. Returns the raw interrupt state of the UARTCTSINTR interrupt.
pub type CTSRMIS_R = crate::register_blocks::BitReader;
/// Field `DCDRMIS` reader - nUARTDCD modem interrupt status. Returns the raw interrupt state of the UARTDCDINTR interrupt.
pub type DCDRMIS_R = crate::register_blocks::BitReader;
/// Field `DSRRMIS` reader - nUARTDSR modem interrupt status. Returns the raw interrupt state of the UARTDSRINTR interrupt.
pub type DSRRMIS_R = crate::register_blocks::BitReader;
/// Field `RXRIS` reader - Receive interrupt status. Returns the raw interrupt state of the UARTRXINTR interrupt.
pub type RXRIS_R = crate::register_blocks::BitReader;
/// Field `TXRIS` reader - Transmit interrupt status. Returns the raw interrupt state of the UARTTXINTR interrupt.
pub type TXRIS_R = crate::register_blocks::BitReader;
/// Field `RTRIS` reader - Receive timeout interrupt status. Returns the raw interrupt state of the UARTRTINTR interrupt. a
pub type RTRIS_R = crate::register_blocks::BitReader;
/// Field `FERIS` reader - Framing error interrupt status. Returns the raw interrupt state of the UARTFEINTR interrupt.
pub type FERIS_R = crate::register_blocks::BitReader;
/// Field `PERIS` reader - Parity error interrupt status. Returns the raw interrupt state of the UARTPEINTR interrupt.
pub type PERIS_R = crate::register_blocks::BitReader;
/// Field `BERIS` reader - Break error interrupt status. Returns the raw interrupt state of the UARTBEINTR interrupt.
pub type BERIS_R = crate::register_blocks::BitReader;
/// Field `OERIS` reader - Overrun error interrupt status. Returns the raw interrupt state of the UARTOEINTR interrupt.
pub type OERIS_R = crate::register_blocks::BitReader;
impl R {
    /// Bit 0 - nUARTRI modem interrupt status. Returns the raw interrupt state of the UARTRIINTR interrupt.
    #[inline(always)]
    pub fn rirmis(&self) -> RIRMIS_R {
        RIRMIS_R::new((self.bits & 1) != 0)
    }
    /// Bit 1 - nUARTCTS modem interrupt status. Returns the raw interrupt state of the UARTCTSINTR interrupt.
    #[inline(always)]
    pub fn ctsrmis(&self) -> CTSRMIS_R {
        CTSRMIS_R::new(((self.bits >> 1) & 1) != 0)
    }
    /// Bit 2 - nUARTDCD modem interrupt status. Returns the raw interrupt state of the UARTDCDINTR interrupt.
    #[inline(always)]
    pub fn dcdrmis(&self) -> DCDRMIS_R {
        DCDRMIS_R::new(((self.bits >> 2) & 1) != 0)
    }
    /// Bit 3 - nUARTDSR modem interrupt status. Returns the raw interrupt state of the UARTDSRINTR interrupt.
    #[inline(always)]
    pub fn dsrrmis(&self) -> DSRRMIS_R {
        DSRRMIS_R::new(((self.bits >> 3) & 1) != 0)
    }
    /// Bit 4 - Receive interrupt status. Returns the raw interrupt state of the UARTRXINTR interrupt.
    #[inline(always)]
    pub fn rxris(&self) -> RXRIS_R {
        RXRIS_R::new(((self.bits >> 4) & 1) != 0)
    }
    /// Bit 5 - Transmit interrupt status. Returns the raw interrupt state of the UARTTXINTR interrupt.
    #[inline(always)]
    pub fn txris(&self) -> TXRIS_R {
        TXRIS_R::new(((self.bits >> 5) & 1) != 0)
    }
    /// Bit 6 - Receive timeout interrupt status. Returns the raw interrupt state of the UARTRTINTR interrupt. a
    #[inline(always)]
    pub fn rtris(&self) -> RTRIS_R {
        RTRIS_R::new(((self.bits >> 6) & 1) != 0)
    }
    /// Bit 7 - Framing error interrupt status. Returns the raw interrupt state of the UARTFEINTR interrupt.
    #[inline(always)]
    pub fn feris(&self) -> FERIS_R {
        FERIS_R::new(((self.bits >> 7) & 1) != 0)
    }
    /// Bit 8 - Parity error interrupt status. Returns the raw interrupt state of the UARTPEINTR interrupt.
    #[inline(always)]
    pub fn peris(&self) -> PERIS_R {
        PERIS_R::new(((self.bits >> 8) & 1) != 0)
    }
    /// Bit 9 - Break error interrupt status. Returns the raw interrupt state of the UARTBEINTR interrupt.
    #[inline(always)]
    pub fn beris(&self) -> BERIS_R {
        BERIS_R::new(((self.bits >> 9) & 1) != 0)
    }
    /// Bit 10 - Overrun error interrupt status. Returns the raw interrupt state of the UARTOEINTR interrupt.
    #[inline(always)]
    pub fn oeris(&self) -> OERIS_R {
        OERIS_R::new(((self.bits >> 10) & 1) != 0)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTRIS")
            .field("oeris", &self.oeris())
            .field("beris", &self.beris())
            .field("peris", &self.peris())
            .field("feris", &self.feris())
            .field("rtris", &self.rtris())
            .field("txris", &self.txris())
            .field("rxris", &self.rxris())
            .field("dsrrmis", &self.dsrrmis())
            .field("dcdrmis", &self.dcdrmis())
            .field("ctsrmis", &self.ctsrmis())
            .field("rirmis", &self.rirmis())
            .finish()
    }
}
/// Raw Interrupt Status Register, UARTRIS  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartris::R`](R). See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTRIS_SPEC;
impl crate::register_blocks::RegisterSpec for UARTRIS_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartris::R`](R) reader structure
impl crate::register_blocks::Readable for UARTRIS_SPEC {}
/// `reset()` method sets UARTRIS to value 0
impl crate::register_blocks::Resettable for UARTRIS_SPEC {}
