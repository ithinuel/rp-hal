/// Register `UARTFBRD` reader
pub type R = crate::register_blocks::R<UARTFBRD_SPEC>;
/// Register `UARTFBRD` writer
pub type W = crate::register_blocks::W<UARTFBRD_SPEC>;
/// Field `BAUD_DIVFRAC` reader - The fractional baud rate divisor. These bits are cleared to 0 on reset.
pub type BAUD_DIVFRAC_R = crate::register_blocks::FieldReader;
/// Field `BAUD_DIVFRAC` writer - The fractional baud rate divisor. These bits are cleared to 0 on reset.
pub type BAUD_DIVFRAC_W<'a, REG> = crate::register_blocks::FieldWriter<'a, REG, 6>;
impl R {
    /// Bits 0:5 - The fractional baud rate divisor. These bits are cleared to 0 on reset.
    #[inline(always)]
    pub fn baud_divfrac(&self) -> BAUD_DIVFRAC_R {
        BAUD_DIVFRAC_R::new((self.bits & 0x3f) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTFBRD")
            .field("baud_divfrac", &self.baud_divfrac())
            .finish()
    }
}
impl W {
    /// Bits 0:5 - The fractional baud rate divisor. These bits are cleared to 0 on reset.
    #[inline(always)]
    pub fn baud_divfrac(&mut self) -> BAUD_DIVFRAC_W<UARTFBRD_SPEC> {
        BAUD_DIVFRAC_W::new(self, 0)
    }
}
/// Fractional Baud Rate Register, UARTFBRD  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartfbrd::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartfbrd::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTFBRD_SPEC;
impl crate::register_blocks::RegisterSpec for UARTFBRD_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartfbrd::R`](R) reader structure
impl crate::register_blocks::Readable for UARTFBRD_SPEC {}
/// `write(|w| ..)` method takes [`uartfbrd::W`](W) writer structure
impl crate::register_blocks::Writable for UARTFBRD_SPEC {
    type Safety = crate::register_blocks::Unsafe;
}
/// `reset()` method sets UARTFBRD to value 0
impl crate::register_blocks::Resettable for UARTFBRD_SPEC {}
