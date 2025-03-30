/// Register `UARTICR` reader
pub type R = crate::register_blocks::R<UARTICR_SPEC>;
/// Register `UARTICR` writer
pub type W = crate::register_blocks::W<UARTICR_SPEC>;
/// Field `RIMIC` reader - nUARTRI modem interrupt clear. Clears the UARTRIINTR interrupt.
pub type RIMIC_R = crate::register_blocks::BitReader;
/// Field `RIMIC` writer - nUARTRI modem interrupt clear. Clears the UARTRIINTR interrupt.
pub type RIMIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `CTSMIC` reader - nUARTCTS modem interrupt clear. Clears the UARTCTSINTR interrupt.
pub type CTSMIC_R = crate::register_blocks::BitReader;
/// Field `CTSMIC` writer - nUARTCTS modem interrupt clear. Clears the UARTCTSINTR interrupt.
pub type CTSMIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `DCDMIC` reader - nUARTDCD modem interrupt clear. Clears the UARTDCDINTR interrupt.
pub type DCDMIC_R = crate::register_blocks::BitReader;
/// Field `DCDMIC` writer - nUARTDCD modem interrupt clear. Clears the UARTDCDINTR interrupt.
pub type DCDMIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `DSRMIC` reader - nUARTDSR modem interrupt clear. Clears the UARTDSRINTR interrupt.
pub type DSRMIC_R = crate::register_blocks::BitReader;
/// Field `DSRMIC` writer - nUARTDSR modem interrupt clear. Clears the UARTDSRINTR interrupt.
pub type DSRMIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `RXIC` reader - Receive interrupt clear. Clears the UARTRXINTR interrupt.
pub type RXIC_R = crate::register_blocks::BitReader;
/// Field `RXIC` writer - Receive interrupt clear. Clears the UARTRXINTR interrupt.
pub type RXIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `TXIC` reader - Transmit interrupt clear. Clears the UARTTXINTR interrupt.
pub type TXIC_R = crate::register_blocks::BitReader;
/// Field `TXIC` writer - Transmit interrupt clear. Clears the UARTTXINTR interrupt.
pub type TXIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `RTIC` reader - Receive timeout interrupt clear. Clears the UARTRTINTR interrupt.
pub type RTIC_R = crate::register_blocks::BitReader;
/// Field `RTIC` writer - Receive timeout interrupt clear. Clears the UARTRTINTR interrupt.
pub type RTIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `FEIC` reader - Framing error interrupt clear. Clears the UARTFEINTR interrupt.
pub type FEIC_R = crate::register_blocks::BitReader;
/// Field `FEIC` writer - Framing error interrupt clear. Clears the UARTFEINTR interrupt.
pub type FEIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `PEIC` reader - Parity error interrupt clear. Clears the UARTPEINTR interrupt.
pub type PEIC_R = crate::register_blocks::BitReader;
/// Field `PEIC` writer - Parity error interrupt clear. Clears the UARTPEINTR interrupt.
pub type PEIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `BEIC` reader - Break error interrupt clear. Clears the UARTBEINTR interrupt.
pub type BEIC_R = crate::register_blocks::BitReader;
/// Field `BEIC` writer - Break error interrupt clear. Clears the UARTBEINTR interrupt.
pub type BEIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
/// Field `OEIC` reader - Overrun error interrupt clear. Clears the UARTOEINTR interrupt.
pub type OEIC_R = crate::register_blocks::BitReader;
/// Field `OEIC` writer - Overrun error interrupt clear. Clears the UARTOEINTR interrupt.
pub type OEIC_W<'a, REG> = crate::register_blocks::BitWriter1C<'a, REG>;
impl R {
    /// Bit 0 - nUARTRI modem interrupt clear. Clears the UARTRIINTR interrupt.
    #[inline(always)]
    pub fn rimic(&self) -> RIMIC_R {
        RIMIC_R::new((self.bits & 1) != 0)
    }
    /// Bit 1 - nUARTCTS modem interrupt clear. Clears the UARTCTSINTR interrupt.
    #[inline(always)]
    pub fn ctsmic(&self) -> CTSMIC_R {
        CTSMIC_R::new(((self.bits >> 1) & 1) != 0)
    }
    /// Bit 2 - nUARTDCD modem interrupt clear. Clears the UARTDCDINTR interrupt.
    #[inline(always)]
    pub fn dcdmic(&self) -> DCDMIC_R {
        DCDMIC_R::new(((self.bits >> 2) & 1) != 0)
    }
    /// Bit 3 - nUARTDSR modem interrupt clear. Clears the UARTDSRINTR interrupt.
    #[inline(always)]
    pub fn dsrmic(&self) -> DSRMIC_R {
        DSRMIC_R::new(((self.bits >> 3) & 1) != 0)
    }
    /// Bit 4 - Receive interrupt clear. Clears the UARTRXINTR interrupt.
    #[inline(always)]
    pub fn rxic(&self) -> RXIC_R {
        RXIC_R::new(((self.bits >> 4) & 1) != 0)
    }
    /// Bit 5 - Transmit interrupt clear. Clears the UARTTXINTR interrupt.
    #[inline(always)]
    pub fn txic(&self) -> TXIC_R {
        TXIC_R::new(((self.bits >> 5) & 1) != 0)
    }
    /// Bit 6 - Receive timeout interrupt clear. Clears the UARTRTINTR interrupt.
    #[inline(always)]
    pub fn rtic(&self) -> RTIC_R {
        RTIC_R::new(((self.bits >> 6) & 1) != 0)
    }
    /// Bit 7 - Framing error interrupt clear. Clears the UARTFEINTR interrupt.
    #[inline(always)]
    pub fn feic(&self) -> FEIC_R {
        FEIC_R::new(((self.bits >> 7) & 1) != 0)
    }
    /// Bit 8 - Parity error interrupt clear. Clears the UARTPEINTR interrupt.
    #[inline(always)]
    pub fn peic(&self) -> PEIC_R {
        PEIC_R::new(((self.bits >> 8) & 1) != 0)
    }
    /// Bit 9 - Break error interrupt clear. Clears the UARTBEINTR interrupt.
    #[inline(always)]
    pub fn beic(&self) -> BEIC_R {
        BEIC_R::new(((self.bits >> 9) & 1) != 0)
    }
    /// Bit 10 - Overrun error interrupt clear. Clears the UARTOEINTR interrupt.
    #[inline(always)]
    pub fn oeic(&self) -> OEIC_R {
        OEIC_R::new(((self.bits >> 10) & 1) != 0)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTICR")
            .field("oeic", &self.oeic())
            .field("beic", &self.beic())
            .field("peic", &self.peic())
            .field("feic", &self.feic())
            .field("rtic", &self.rtic())
            .field("txic", &self.txic())
            .field("rxic", &self.rxic())
            .field("dsrmic", &self.dsrmic())
            .field("dcdmic", &self.dcdmic())
            .field("ctsmic", &self.ctsmic())
            .field("rimic", &self.rimic())
            .finish()
    }
}
impl W {
    /// Bit 0 - nUARTRI modem interrupt clear. Clears the UARTRIINTR interrupt.
    #[inline(always)]
    pub fn rimic(&mut self) -> RIMIC_W<UARTICR_SPEC> {
        RIMIC_W::new(self, 0)
    }
    /// Bit 1 - nUARTCTS modem interrupt clear. Clears the UARTCTSINTR interrupt.
    #[inline(always)]
    pub fn ctsmic(&mut self) -> CTSMIC_W<UARTICR_SPEC> {
        CTSMIC_W::new(self, 1)
    }
    /// Bit 2 - nUARTDCD modem interrupt clear. Clears the UARTDCDINTR interrupt.
    #[inline(always)]
    pub fn dcdmic(&mut self) -> DCDMIC_W<UARTICR_SPEC> {
        DCDMIC_W::new(self, 2)
    }
    /// Bit 3 - nUARTDSR modem interrupt clear. Clears the UARTDSRINTR interrupt.
    #[inline(always)]
    pub fn dsrmic(&mut self) -> DSRMIC_W<UARTICR_SPEC> {
        DSRMIC_W::new(self, 3)
    }
    /// Bit 4 - Receive interrupt clear. Clears the UARTRXINTR interrupt.
    #[inline(always)]
    pub fn rxic(&mut self) -> RXIC_W<UARTICR_SPEC> {
        RXIC_W::new(self, 4)
    }
    /// Bit 5 - Transmit interrupt clear. Clears the UARTTXINTR interrupt.
    #[inline(always)]
    pub fn txic(&mut self) -> TXIC_W<UARTICR_SPEC> {
        TXIC_W::new(self, 5)
    }
    /// Bit 6 - Receive timeout interrupt clear. Clears the UARTRTINTR interrupt.
    #[inline(always)]
    pub fn rtic(&mut self) -> RTIC_W<UARTICR_SPEC> {
        RTIC_W::new(self, 6)
    }
    /// Bit 7 - Framing error interrupt clear. Clears the UARTFEINTR interrupt.
    #[inline(always)]
    pub fn feic(&mut self) -> FEIC_W<UARTICR_SPEC> {
        FEIC_W::new(self, 7)
    }
    /// Bit 8 - Parity error interrupt clear. Clears the UARTPEINTR interrupt.
    #[inline(always)]
    pub fn peic(&mut self) -> PEIC_W<UARTICR_SPEC> {
        PEIC_W::new(self, 8)
    }
    /// Bit 9 - Break error interrupt clear. Clears the UARTBEINTR interrupt.
    #[inline(always)]
    pub fn beic(&mut self) -> BEIC_W<UARTICR_SPEC> {
        BEIC_W::new(self, 9)
    }
    /// Bit 10 - Overrun error interrupt clear. Clears the UARTOEINTR interrupt.
    #[inline(always)]
    pub fn oeic(&mut self) -> OEIC_W<UARTICR_SPEC> {
        OEIC_W::new(self, 10)
    }
}
/// Interrupt Clear Register, UARTICR  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uarticr::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uarticr::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTICR_SPEC;
impl crate::register_blocks::RegisterSpec for UARTICR_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uarticr::R`](R) reader structure
impl crate::register_blocks::Readable for UARTICR_SPEC {}
/// `write(|w| ..)` method takes [`uarticr::W`](W) writer structure
impl crate::register_blocks::Writable for UARTICR_SPEC {
    type Safety = crate::register_blocks::Unsafe;
    const ONE_TO_MODIFY_FIELDS_BITMAP: u32 = 0x07ff;
}
/// `reset()` method sets UARTICR to value 0
impl crate::register_blocks::Resettable for UARTICR_SPEC {}
