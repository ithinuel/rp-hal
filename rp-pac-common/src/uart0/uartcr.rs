use crate::register_blocks::{self, BitReader, BitWriter};

/// Register `UARTCR` reader
pub type R = register_blocks::R<UARTCR_SPEC>;
/// Register `UARTCR` writer
pub type W = register_blocks::W<UARTCR_SPEC>;
/// Field `UARTEN` reader - UART enable:
/// - 0 UART is disabled.
///   If the UART is disabled in the middle of transmission or reception, it completes the current
///   character before stopping.
/// - 1 the UART is enabled.
///   Data transmission and reception occurs for either UART signals or SIR signals depending on the
///   setting of the SIREN bit.
pub type UARTEN_R = BitReader;
/// Field `UARTEN` writer - UART enable:
/// - 0 UART is disabled. If the UART is disabled in the middle of transmission or reception, it
///   completes the current character before stopping.
/// - 1 the UART is enabled. Data transmission and reception occurs for either UART signals or SIR
///   signals depending on the setting of the SIREN bit.
pub type UARTEN_W<'a, REG> = BitWriter<'a, REG>;
/// Field `SIREN` reader - SIR enable:
/// - 0 IrDA SIR ENDEC is disabled. nSIROUT remains LOW (no light pulse generated), and signal
///   transitions on SIRIN have no effect.
/// - 1 IrDA SIR ENDEC is enabled. Data is transmitted and received on nSIROUT and SIRIN. UARTTXD
///   remains HIGH, in the marking state.
///
/// Signal transitions on UARTRXD or modem status inputs have no effect.
/// This bit has no effect if the UARTEN bit disables the UART.
pub type SIREN_R = BitReader;
/// Field `SIREN` writer - SIR enable:
/// - 0 IrDA SIR ENDEC is disabled.
///   nSIROUT remains LOW (no light pulse generated), and signal transitions on SIRIN have no effect.
/// - 1 IrDA SIR ENDEC is enabled.
///   Data is transmitted and received on nSIROUT and SIRIN.
///   UARTTXD remains HIGH, in the marking state. Signal transitions on UARTRXD or modem status inputs
///   have no effect. This bit has no effect if the UARTEN bit disables the UART.
pub type SIREN_W<'a, REG> = BitWriter<'a, REG>;
/// Field `SIRLP` reader - SIR low-power IrDA mode.
/// This bit selects the IrDA encoding mode.
///
/// If this bit is cleared to 0, low-level bits are transmitted as an active high pulse with a width of 3 / 16th of the bit period.
/// If this bit is set to 1, low-level bits are transmitted with a pulse width that is 3 times the period of the IrLPBaud16 input signal, regardless of the selected bit rate. Setting this bit uses less power, but might reduce transmission distances.
pub type SIRLP_R = BitReader;
/// Field `SIRLP` writer - SIR low-power IrDA mode. This bit selects the IrDA encoding mode.
/// If this bit is cleared to 0, low-level bits are transmitted as an active high pulse with a width of 3 / 16th of the bit period.
/// If this bit is set to 1, low-level bits are transmitted with a pulse width that is 3 times the period of the IrLPBaud16 input signal, regardless of the selected bit rate. Setting this bit uses less power, but might reduce transmission distances.
pub type SIRLP_W<'a, REG> = BitWriter<'a, REG>;
/// Field `LBE` reader - Loopback enable.
/// If this bit is set to 1 and the SIREN bit is set to 1 and the SIRTEST bit in the Test Control Register, UARTTCR is set to 1, then the nSIROUT path is inverted, and fed through to the SIRIN path. The SIRTEST bit in the test register must be set to 1 to override the normal half-duplex SIR operation. This must be the requirement for accessing the test registers during normal operation, and SIRTEST must be cleared to 0 when loopback testing is finished. This feature reduces the amount of external coupling required during system test.
/// If this bit is set to 1, and the SIRTEST bit is set to 0, the UARTTXD path is fed through to the UARTRXD path. In either SIR mode or UART mode, when this bit is set, the modem outputs are also fed through to the modem inputs. This bit is cleared to 0 on reset, to disable loopback.
pub type LBE_R = BitReader;
/// Field `LBE` writer - Loopback enable.
/// If this bit is set to 1 and the SIREN bit is set to 1 and the SIRTEST bit in the Test Control Register, UARTTCR is set to 1, then the nSIROUT path is inverted, and fed through to the SIRIN path. The SIRTEST bit in the test register must be set to 1 to override the normal half-duplex SIR operation. This must be the requirement for accessing the test registers during normal operation, and SIRTEST must be cleared to 0 when loopback testing is finished. This feature reduces the amount of external coupling required during system test.
/// If this bit is set to 1, and the SIRTEST bit is set to 0, the UARTTXD path is fed through to the UARTRXD path. In either SIR mode or UART mode, when this bit is set, the modem outputs are also fed through to the modem inputs. This bit is cleared to 0 on reset, to disable loopback.
pub type LBE_W<'a, REG> = BitWriter<'a, REG>;
/// Field `TXE` reader - Transmit enable.
/// If this bit is set to 1, the transmit section of the UART is enabled. Data transmission occurs for either UART signals, or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of transmission, it completes the current character before stopping.
pub type TXE_R = BitReader;
/// Field `TXE` writer - Transmit enable.
/// If this bit is set to 1, the transmit section of the UART is enabled. Data transmission occurs for either UART signals, or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of transmission, it completes the current character before stopping.
pub type TXE_W<'a, REG> = BitWriter<'a, REG>;
/// Field `RXE` reader - Receive enable.
/// If this bit is set to 1, the receive section of the UART is enabled. Data reception occurs for either UART signals or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of reception, it completes the current character before stopping.
pub type RXE_R = BitReader;
/// Field `RXE` writer - Receive enable.
/// If this bit is set to 1, the receive section of the UART is enabled. Data reception occurs for either UART signals or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of reception, it completes the current character before stopping.
pub type RXE_W<'a, REG> = BitWriter<'a, REG>;
/// Field `DTR` reader - Data transmit ready. This bit is the complement of the UART data transmit ready, nUARTDTR, modem status output. That is, when the bit is programmed to a 1 then nUARTDTR is LOW.
pub type DTR_R = BitReader;
/// Field `DTR` writer - Data transmit ready. This bit is the complement of the UART data transmit ready, nUARTDTR, modem status output. That is, when the bit is programmed to a 1 then nUARTDTR is LOW.
pub type DTR_W<'a, REG> = BitWriter<'a, REG>;
/// Field `RTS` reader - Request to send. This bit is the complement of the UART request to send, nUARTRTS, modem status output. That is, when the bit is programmed to a 1 then nUARTRTS is LOW.
pub type RTS_R = BitReader;
/// Field `RTS` writer - Request to send. This bit is the complement of the UART request to send, nUARTRTS, modem status output. That is, when the bit is programmed to a 1 then nUARTRTS is LOW.
pub type RTS_W<'a, REG> = BitWriter<'a, REG>;
/// Field `OUT1` reader - This bit is the complement of the UART Out1 (nUARTOut1) modem status output. That is, when the bit is programmed to a 1 the output is 0. For DTE this can be used as Data Carrier Detect (DCD).
pub type OUT1_R = BitReader;
/// Field `OUT1` writer - This bit is the complement of the UART Out1 (nUARTOut1) modem status output. That is, when the bit is programmed to a 1 the output is 0. For DTE this can be used as Data Carrier Detect (DCD).
pub type OUT1_W<'a, REG> = BitWriter<'a, REG>;
/// Field `OUT2` reader - This bit is the complement of the UART Out2 (nUARTOut2) modem status output. That is, when the bit is programmed to a 1, the output is 0. For DTE this can be used as Ring Indicator (RI).
pub type OUT2_R = BitReader;
/// Field `OUT2` writer - This bit is the complement of the UART Out2 (nUARTOut2) modem status output. That is, when the bit is programmed to a 1, the output is 0. For DTE this can be used as Ring Indicator (RI).
pub type OUT2_W<'a, REG> = BitWriter<'a, REG>;
/// Field `RTSEN` reader - RTS hardware flow control enable.
/// If this bit is set to 1, RTS hardware flow control is enabled. Data is only requested when there is space in the receive FIFO for it to be received.
pub type RTSEN_R = BitReader;
/// Field `RTSEN` writer - RTS hardware flow control enable.
/// If this bit is set to 1, RTS hardware flow control is enabled. Data is only requested when there is space in the receive FIFO for it to be received.
pub type RTSEN_W<'a, REG> = BitWriter<'a, REG>;
/// Field `CTSEN` reader - CTS hardware flow control enable.
/// If this bit is set to 1, CTS hardware flow control is enabled. Data is only transmitted when the nUARTCTS signal is asserted.
pub type CTSEN_R = BitReader;
/// Field `CTSEN` writer - CTS hardware flow control enable.
/// If this bit is set to 1, CTS hardware flow control is enabled. Data is only transmitted when the nUARTCTS signal is asserted.
pub type CTSEN_W<'a, REG> = BitWriter<'a, REG>;
impl R {
    /// Bit 0 - UART enable:
    /// - 0 UART is disabled. If the UART is disabled in the middle of transmission or reception, it completes the current character before stopping.
    /// - 1 the UART is enabled. Data transmission and reception occurs for either UART signals or SIR signals depending on the setting of the SIREN bit.
    #[inline(always)]
    pub fn uarten(&self) -> UARTEN_R {
        UARTEN_R::new((self.bits & 1) != 0)
    }
    /// Bit 1 - SIR enable:
    /// - 0 IrDA SIR ENDEC is disabled. nSIROUT remains LOW (no light pulse generated), and signal transitions on SIRIN have no effect.
    /// - 1 IrDA SIR ENDEC is enabled. Data is transmitted and received on nSIROUT and SIRIN. UARTTXD remains HIGH, in the marking state. Signal transitions on UARTRXD or modem status inputs have no effect. This bit has no effect if the UARTEN bit disables the UART.
    #[inline(always)]
    pub fn siren(&self) -> SIREN_R {
        SIREN_R::new(((self.bits >> 1) & 1) != 0)
    }
    /// Bit 2 - SIR low-power IrDA mode. This bit selects the IrDA encoding mode.
    /// If this bit is cleared to 0, low-level bits are transmitted as an active high pulse with a width of 3 / 16th of the bit period.
    /// If this bit is set to 1, low-level bits are transmitted with a pulse width that is 3 times the period of the IrLPBaud16 input signal, regardless of the selected bit rate. Setting this bit uses less power, but might reduce transmission distances.
    #[inline(always)]
    pub fn sirlp(&self) -> SIRLP_R {
        SIRLP_R::new(((self.bits >> 2) & 1) != 0)
    }
    /// Bit 7 - Loopback enable.
    /// If this bit is set to 1 and the SIREN bit is set to 1 and the SIRTEST bit in the Test Control Register, UARTTCR is set to 1, then the nSIROUT path is inverted, and fed through to the SIRIN path. The SIRTEST bit in the test register must be set to 1 to override the normal half-duplex SIR operation. This must be the requirement for accessing the test registers during normal operation, and SIRTEST must be cleared to 0 when loopback testing is finished. This feature reduces the amount of external coupling required during system test.
    /// If this bit is set to 1, and the SIRTEST bit is set to 0, the UARTTXD path is fed through to the UARTRXD path. In either SIR mode or UART mode, when this bit is set, the modem outputs are also fed through to the modem inputs. This bit is cleared to 0 on reset, to disable loopback.
    #[inline(always)]
    pub fn lbe(&self) -> LBE_R {
        LBE_R::new(((self.bits >> 7) & 1) != 0)
    }
    /// Bit 8 - Transmit enable.
    /// If this bit is set to 1, the transmit section of the UART is enabled. Data transmission occurs for either UART signals, or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of transmission, it completes the current character before stopping.
    #[inline(always)]
    pub fn txe(&self) -> TXE_R {
        TXE_R::new(((self.bits >> 8) & 1) != 0)
    }
    /// Bit 9 - Receive enable.
    /// If this bit is set to 1, the receive section of the UART is enabled. Data reception occurs for either UART signals or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of reception, it completes the current character before stopping.
    #[inline(always)]
    pub fn rxe(&self) -> RXE_R {
        RXE_R::new(((self.bits >> 9) & 1) != 0)
    }
    /// Bit 10 - Data transmit ready. This bit is the complement of the UART data transmit ready, nUARTDTR, modem status output. That is, when the bit is programmed to a 1 then nUARTDTR is LOW.
    #[inline(always)]
    pub fn dtr(&self) -> DTR_R {
        DTR_R::new(((self.bits >> 10) & 1) != 0)
    }
    /// Bit 11 - Request to send. This bit is the complement of the UART request to send, nUARTRTS, modem status output. That is, when the bit is programmed to a 1 then nUARTRTS is LOW.
    #[inline(always)]
    pub fn rts(&self) -> RTS_R {
        RTS_R::new(((self.bits >> 11) & 1) != 0)
    }
    /// Bit 12 - This bit is the complement of the UART Out1 (nUARTOut1) modem status output. That is, when the bit is programmed to a 1 the output is 0. For DTE this can be used as Data Carrier Detect (DCD).
    #[inline(always)]
    pub fn out1(&self) -> OUT1_R {
        OUT1_R::new(((self.bits >> 12) & 1) != 0)
    }
    /// Bit 13 - This bit is the complement of the UART Out2 (nUARTOut2) modem status output. That is, when the bit is programmed to a 1, the output is 0. For DTE this can be used as Ring Indicator (RI).
    #[inline(always)]
    pub fn out2(&self) -> OUT2_R {
        OUT2_R::new(((self.bits >> 13) & 1) != 0)
    }
    /// Bit 14 - RTS hardware flow control enable.
    /// If this bit is set to 1, RTS hardware flow control is enabled. Data is only requested when there is space in the receive FIFO for it to be received.
    #[inline(always)]
    pub fn rtsen(&self) -> RTSEN_R {
        RTSEN_R::new(((self.bits >> 14) & 1) != 0)
    }
    /// Bit 15 - CTS hardware flow control enable.
    /// If this bit is set to 1, CTS hardware flow control is enabled. Data is only transmitted when the nUARTCTS signal is asserted.
    #[inline(always)]
    pub fn ctsen(&self) -> CTSEN_R {
        CTSEN_R::new(((self.bits >> 15) & 1) != 0)
    }
}
impl crate::register_blocks::generic::raw::MyRDebug for UARTCR_SPEC {
    fn fmt(this: &R, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTCR")
            .field("ctsen", &this.ctsen())
            .field("rtsen", &this.rtsen())
            .field("out2", &this.out2())
            .field("out1", &this.out1())
            .field("rts", &this.rts())
            .field("dtr", &this.dtr())
            .field("rxe", &this.rxe())
            .field("txe", &this.txe())
            .field("lbe", &this.lbe())
            .field("sirlp", &this.sirlp())
            .field("siren", &this.siren())
            .field("uarten", &this.uarten())
            .finish()
    }
}
impl W {
    /// Bit 0 - UART enable:
    /// - 0 UART is disabled.
    ///   If the UART is disabled in the middle of transmission or reception, it completes the current
    ///   character before stopping.
    /// - 1 the UART is enabled.
    ///   Data transmission and reception occurs for either UART signals or SIR signals depending on
    ///   the setting of the SIREN bit.
    #[inline(always)]
    pub fn uarten(&mut self) -> UARTEN_W<UARTCR_SPEC> {
        UARTEN_W::new(self, 0)
    }
    /// Bit 1 - SIR enable:
    /// - 0 IrDA SIR ENDEC is disabled. nSIROUT remains LOW (no light pulse generated), and signal transitions on SIRIN have no effect.
    /// - 1 IrDA SIR ENDEC is enabled. Data is transmitted and received on nSIROUT and SIRIN. UARTTXD remains HIGH, in the marking state. Signal transitions on UARTRXD or modem status inputs have no effect. This bit has no effect if the UARTEN bit disables the UART.
    #[inline(always)]
    pub fn siren(&mut self) -> SIREN_W<UARTCR_SPEC> {
        SIREN_W::new(self, 1)
    }
    /// Bit 2 - SIR low-power IrDA mode. This bit selects the IrDA encoding mode.
    /// If this bit is cleared to 0, low-level bits are transmitted as an active high pulse with a width of 3 / 16th of the bit period.
    /// If this bit is set to 1, low-level bits are transmitted with a pulse width that is 3 times the period of the IrLPBaud16 input signal, regardless of the selected bit rate. Setting this bit uses less power, but might reduce transmission distances.
    #[inline(always)]
    pub fn sirlp(&mut self) -> SIRLP_W<UARTCR_SPEC> {
        SIRLP_W::new(self, 2)
    }
    /// Bit 7 - Loopback enable.
    /// If this bit is set to 1 and the SIREN bit is set to 1 and the SIRTEST bit in the Test Control Register, UARTTCR is set to 1, then the nSIROUT path is inverted, and fed through to the SIRIN path. The SIRTEST bit in the test register must be set to 1 to override the normal half-duplex SIR operation. This must be the requirement for accessing the test registers during normal operation, and SIRTEST must be cleared to 0 when loopback testing is finished. This feature reduces the amount of external coupling required during system test.
    /// If this bit is set to 1, and the SIRTEST bit is set to 0, the UARTTXD path is fed through to the UARTRXD path. In either SIR mode or UART mode, when this bit is set, the modem outputs are also fed through to the modem inputs. This bit is cleared to 0 on reset, to disable loopback.
    #[inline(always)]
    pub fn lbe(&mut self) -> LBE_W<UARTCR_SPEC> {
        LBE_W::new(self, 7)
    }
    /// Bit 8 - Transmit enable.
    /// If this bit is set to 1, the transmit section of the UART is enabled. Data transmission occurs for either UART signals, or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of transmission, it completes the current character before stopping.
    #[inline(always)]
    pub fn txe(&mut self) -> TXE_W<UARTCR_SPEC> {
        TXE_W::new(self, 8)
    }
    /// Bit 9 - Receive enable.
    /// If this bit is set to 1, the receive section of the UART is enabled. Data reception occurs for either UART signals or SIR signals depending on the setting of the SIREN bit. When the UART is disabled in the middle of reception, it completes the current character before stopping.
    #[inline(always)]
    pub fn rxe(&mut self) -> RXE_W<UARTCR_SPEC> {
        RXE_W::new(self, 9)
    }
    /// Bit 10 - Data transmit ready. This bit is the complement of the UART data transmit ready, nUARTDTR, modem status output. That is, when the bit is programmed to a 1 then nUARTDTR is LOW.
    #[inline(always)]
    pub fn dtr(&mut self) -> DTR_W<UARTCR_SPEC> {
        DTR_W::new(self, 10)
    }
    /// Bit 11 - Request to send. This bit is the complement of the UART request to send, nUARTRTS, modem status output. That is, when the bit is programmed to a 1 then nUARTRTS is LOW.
    #[inline(always)]
    pub fn rts(&mut self) -> RTS_W<UARTCR_SPEC> {
        RTS_W::new(self, 11)
    }
    /// Bit 12 - This bit is the complement of the UART Out1 (nUARTOut1) modem status output. That is, when the bit is programmed to a 1 the output is 0. For DTE this can be used as Data Carrier Detect (DCD).
    #[inline(always)]
    pub fn out1(&mut self) -> OUT1_W<UARTCR_SPEC> {
        OUT1_W::new(self, 12)
    }
    /// Bit 13 - This bit is the complement of the UART Out2 (nUARTOut2) modem status output. That is, when the bit is programmed to a 1, the output is 0. For DTE this can be used as Ring Indicator (RI).
    #[inline(always)]
    pub fn out2(&mut self) -> OUT2_W<UARTCR_SPEC> {
        OUT2_W::new(self, 13)
    }
    /// Bit 14 - RTS hardware flow control enable.
    /// If this bit is set to 1, RTS hardware flow control is enabled. Data is only requested when there is space in the receive FIFO for it to be received.
    #[inline(always)]
    pub fn rtsen(&mut self) -> RTSEN_W<UARTCR_SPEC> {
        RTSEN_W::new(self, 14)
    }
    /// Bit 15 - CTS hardware flow control enable.
    /// If this bit is set to 1, CTS hardware flow control is enabled. Data is only transmitted when the nUARTCTS signal is asserted.
    #[inline(always)]
    pub fn ctsen(&mut self) -> CTSEN_W<UARTCR_SPEC> {
        CTSEN_W::new(self, 15)
    }
}
/// Control Register, UARTCR  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartcr::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartcr::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTCR_SPEC;
impl crate::register_blocks::RegisterSpec for UARTCR_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartcr::R`](R) reader structure
impl crate::register_blocks::Readable for UARTCR_SPEC {}
/// `write(|w| ..)` method takes [`uartcr::W`](W) writer structure
impl crate::register_blocks::Writable for UARTCR_SPEC {
    type Safety = crate::register_blocks::Unsafe;
}
/// `reset()` method sets UARTCR to value 0x0300
impl crate::register_blocks::Resettable for UARTCR_SPEC {
    const RESET_VALUE: u32 = 0x0300;
}
