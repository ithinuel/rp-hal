/// Register `UARTFR` reader
pub type R = crate::register_blocks::R<UARTFR_SPEC>;
/// Field `CTS` reader - Clear to send. This bit is the complement of the UART clear to send, nUARTCTS, modem status input. That is, the bit is 1 when nUARTCTS is LOW.
pub type CTS_R = crate::register_blocks::BitReader;
/// Field `DSR` reader - Data set ready. This bit is the complement of the UART data set ready, nUARTDSR, modem status input. That is, the bit is 1 when nUARTDSR is LOW.
pub type DSR_R = crate::register_blocks::BitReader;
/// Field `DCD` reader - Data carrier detect. This bit is the complement of the UART data carrier detect, nUARTDCD, modem status input. That is, the bit is 1 when nUARTDCD is LOW.
pub type DCD_R = crate::register_blocks::BitReader;
/// Field `BUSY` reader - UART busy. If this bit is set to 1, the UART is busy transmitting data. This bit remains set until the complete byte, including all the stop bits, has been sent from the shift register. This bit is set as soon as the transmit FIFO becomes non-empty, regardless of whether the UART is enabled or not.
pub type BUSY_R = crate::register_blocks::BitReader;
/// Field `RXFE` reader - Receive FIFO empty. The meaning of this bit depends on the state of the FEN bit in the UARTLCR_H Register. If the FIFO is disabled, this bit is set when the receive holding register is empty. If the FIFO is enabled, the RXFE bit is set when the receive FIFO is empty.
pub type RXFE_R = crate::register_blocks::BitReader;
/// Field `TXFF` reader - Transmit FIFO full. The meaning of this bit depends on the state of the FEN bit in the UARTLCR_H Register. If the FIFO is disabled, this bit is set when the transmit holding register is full. If the FIFO is enabled, the TXFF bit is set when the transmit FIFO is full.
pub type TXFF_R = crate::register_blocks::BitReader;
/// Field `RXFF` reader - Receive FIFO full. The meaning of this bit depends on the state of the FEN bit in the UARTLCR_H Register. If the FIFO is disabled, this bit is set when the receive holding register is full. If the FIFO is enabled, the RXFF bit is set when the receive FIFO is full.
pub type RXFF_R = crate::register_blocks::BitReader;
/// Field `TXFE` reader - Transmit FIFO empty. The meaning of this bit depends on the state of the FEN bit in the Line Control Register, UARTLCR_H. If the FIFO is disabled, this bit is set when the transmit holding register is empty. If the FIFO is enabled, the TXFE bit is set when the transmit FIFO is empty. This bit does not indicate if there is data in the transmit shift register.
pub type TXFE_R = crate::register_blocks::BitReader;
/// Field `RI` reader - Ring indicator. This bit is the complement of the UART ring indicator, nUARTRI, modem status input. That is, the bit is 1 when nUARTRI is LOW.
pub type RI_R = crate::register_blocks::BitReader;
impl R {
    /// Bit 0 - Clear to send. This bit is the complement of the UART clear to send, nUARTCTS, modem status input. That is, the bit is 1 when nUARTCTS is LOW.
    #[inline(always)]
    pub fn cts(&self) -> CTS_R {
        CTS_R::new((self.bits & 1) != 0)
    }
    /// Bit 1 - Data set ready. This bit is the complement of the UART data set ready, nUARTDSR, modem status input. That is, the bit is 1 when nUARTDSR is LOW.
    #[inline(always)]
    pub fn dsr(&self) -> DSR_R {
        DSR_R::new(((self.bits >> 1) & 1) != 0)
    }
    /// Bit 2 - Data carrier detect. This bit is the complement of the UART data carrier detect, nUARTDCD, modem status input. That is, the bit is 1 when nUARTDCD is LOW.
    #[inline(always)]
    pub fn dcd(&self) -> DCD_R {
        DCD_R::new(((self.bits >> 2) & 1) != 0)
    }
    /// Bit 3 - UART busy. If this bit is set to 1, the UART is busy transmitting data. This bit remains set until the complete byte, including all the stop bits, has been sent from the shift register. This bit is set as soon as the transmit FIFO becomes non-empty, regardless of whether the UART is enabled or not.
    #[inline(always)]
    pub fn busy(&self) -> BUSY_R {
        BUSY_R::new(((self.bits >> 3) & 1) != 0)
    }
    /// Bit 4 - Receive FIFO empty. The meaning of this bit depends on the state of the FEN bit in the UARTLCR_H Register. If the FIFO is disabled, this bit is set when the receive holding register is empty. If the FIFO is enabled, the RXFE bit is set when the receive FIFO is empty.
    #[inline(always)]
    pub fn rxfe(&self) -> RXFE_R {
        RXFE_R::new(((self.bits >> 4) & 1) != 0)
    }
    /// Bit 5 - Transmit FIFO full. The meaning of this bit depends on the state of the FEN bit in the UARTLCR_H Register. If the FIFO is disabled, this bit is set when the transmit holding register is full. If the FIFO is enabled, the TXFF bit is set when the transmit FIFO is full.
    #[inline(always)]
    pub fn txff(&self) -> TXFF_R {
        TXFF_R::new(((self.bits >> 5) & 1) != 0)
    }
    /// Bit 6 - Receive FIFO full. The meaning of this bit depends on the state of the FEN bit in the UARTLCR_H Register. If the FIFO is disabled, this bit is set when the receive holding register is full. If the FIFO is enabled, the RXFF bit is set when the receive FIFO is full.
    #[inline(always)]
    pub fn rxff(&self) -> RXFF_R {
        RXFF_R::new(((self.bits >> 6) & 1) != 0)
    }
    /// Bit 7 - Transmit FIFO empty. The meaning of this bit depends on the state of the FEN bit in the Line Control Register, UARTLCR_H. If the FIFO is disabled, this bit is set when the transmit holding register is empty. If the FIFO is enabled, the TXFE bit is set when the transmit FIFO is empty. This bit does not indicate if there is data in the transmit shift register.
    #[inline(always)]
    pub fn txfe(&self) -> TXFE_R {
        TXFE_R::new(((self.bits >> 7) & 1) != 0)
    }
    /// Bit 8 - Ring indicator. This bit is the complement of the UART ring indicator, nUARTRI, modem status input. That is, the bit is 1 when nUARTRI is LOW.
    #[inline(always)]
    pub fn ri(&self) -> RI_R {
        RI_R::new(((self.bits >> 8) & 1) != 0)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTFR")
            .field("ri", &self.ri())
            .field("txfe", &self.txfe())
            .field("rxff", &self.rxff())
            .field("txff", &self.txff())
            .field("rxfe", &self.rxfe())
            .field("busy", &self.busy())
            .field("dcd", &self.dcd())
            .field("dsr", &self.dsr())
            .field("cts", &self.cts())
            .finish()
    }
}
/// Flag Register, UARTFR  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartfr::R`](R). See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTFR_SPEC;
impl crate::register_blocks::RegisterSpec for UARTFR_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartfr::R`](R) reader structure
impl crate::register_blocks::Readable for UARTFR_SPEC {}
/// `reset()` method sets UARTFR to value 0x90
impl crate::register_blocks::Resettable for UARTFR_SPEC {
    const RESET_VALUE: u32 = 0x90;
}
