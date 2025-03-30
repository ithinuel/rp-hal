/// Register `UARTILPR` reader
pub type R = crate::register_blocks::R<UARTILPR_SPEC>;
/// Register `UARTILPR` writer
pub type W = crate::register_blocks::W<UARTILPR_SPEC>;
/// Field `ILPDVSR` reader - 8-bit low-power divisor value. These bits are cleared to 0 at reset.
pub type ILPDVSR_R = crate::register_blocks::FieldReader;
/// Field `ILPDVSR` writer - 8-bit low-power divisor value. These bits are cleared to 0 at reset.
pub type ILPDVSR_W<'a, REG> = crate::register_blocks::FieldWriter<'a, REG, 8>;
impl R {
    /// Bits 0:7 - 8-bit low-power divisor value. These bits are cleared to 0 at reset.
    #[inline(always)]
    pub fn ilpdvsr(&self) -> ILPDVSR_R {
        ILPDVSR_R::new((self.bits & 0xff) as u8)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTILPR")
            .field("ilpdvsr", &self.ilpdvsr())
            .finish()
    }
}
impl W {
    /// Bits 0:7 - 8-bit low-power divisor value. These bits are cleared to 0 at reset.
    #[inline(always)]
    pub fn ilpdvsr(&mut self) -> ILPDVSR_W<UARTILPR_SPEC> {
        ILPDVSR_W::new(self, 0)
    }
}
/// IrDA Low-Power Counter Register, UARTILPR  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartilpr::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartilpr::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTILPR_SPEC;
impl crate::register_blocks::RegisterSpec for UARTILPR_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartilpr::R`](R) reader structure
impl crate::register_blocks::Readable for UARTILPR_SPEC {}
/// `write(|w| ..)` method takes [`uartilpr::W`](W) writer structure
impl crate::register_blocks::Writable for UARTILPR_SPEC {
    type Safety = crate::register_blocks::Unsafe;
}
/// `reset()` method sets UARTILPR to value 0
impl crate::register_blocks::Resettable for UARTILPR_SPEC {}
