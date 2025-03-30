/// Register `UARTIBRD` reader
pub type R = crate::register_blocks::R<UARTIBRD_SPEC>;
/// Register `UARTIBRD` writer
pub type W = crate::register_blocks::W<UARTIBRD_SPEC>;
/// Field `BAUD_DIVINT` reader - The integer baud rate divisor. These bits are cleared to 0 on reset.
pub type BAUD_DIVINT_R = crate::register_blocks::FieldReader<u16>;
/// Field `BAUD_DIVINT` writer - The integer baud rate divisor. These bits are cleared to 0 on reset.
pub type BAUD_DIVINT_W<'a, REG> = crate::register_blocks::FieldWriter<'a, REG, 16, u16>;
impl R {
    /// Bits 0:15 - The integer baud rate divisor. These bits are cleared to 0 on reset.
    #[inline(always)]
    pub fn baud_divint(&self) -> BAUD_DIVINT_R {
        BAUD_DIVINT_R::new((self.bits & 0xffff) as u16)
    }
}
impl core::fmt::Debug for R {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("UARTIBRD")
            .field("baud_divint", &self.baud_divint())
            .finish()
    }
}
impl W {
    /// Bits 0:15 - The integer baud rate divisor. These bits are cleared to 0 on reset.
    #[inline(always)]
    pub fn baud_divint(&mut self) -> BAUD_DIVINT_W<UARTIBRD_SPEC> {
        BAUD_DIVINT_W::new(self, 0)
    }
}
/// Integer Baud Rate Register, UARTIBRD  /// /// You can [`read`](crate::register_blocks::Reg::read) this register and get [`uartibrd::R`](R). You can [`reset`](crate::register_blocks::Reg::reset), [`write`](crate::register_blocks::Reg::write), [`write_with_zero`](crate::register_blocks::Reg::write_with_zero) this register using [`uartibrd::W`](W). You can also [`modify`](crate::register_blocks::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).
pub struct UARTIBRD_SPEC;
impl crate::register_blocks::RegisterSpec for UARTIBRD_SPEC {
    type Ux = u32;
}
/// `read()` method returns [`uartibrd::R`](R) reader structure
impl crate::register_blocks::Readable for UARTIBRD_SPEC {}
/// `write(|w| ..)` method takes [`uartibrd::W`](W) writer structure
impl crate::register_blocks::Writable for UARTIBRD_SPEC {
    type Safety = crate::register_blocks::Unsafe;
}
/// `reset()` method sets UARTIBRD to value 0
impl crate::register_blocks::Resettable for UARTIBRD_SPEC {}
