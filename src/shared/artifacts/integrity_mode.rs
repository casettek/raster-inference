use std::cell::Cell;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub enum RasterIntegrityMode {
    #[default]
    Verified,
    #[cfg(feature = "unchecked-raster-integrity")]
    UncheckedTestOnly,
}

impl RasterIntegrityMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::UncheckedTestOnly => "unchecked-test-only",
        }
    }

    pub fn is_unchecked_test_only(self) -> bool {
        match self {
            Self::Verified => false,
            #[cfg(feature = "unchecked-raster-integrity")]
            Self::UncheckedTestOnly => true,
        }
    }
}

thread_local! {
    static RASTER_INTEGRITY_MODE: Cell<RasterIntegrityMode> =
        const { Cell::new(RasterIntegrityMode::Verified) };
}

pub fn current_raster_integrity_mode() -> RasterIntegrityMode {
    RASTER_INTEGRITY_MODE.with(Cell::get)
}

pub fn raster_integrity_is_unchecked() -> bool {
    current_raster_integrity_mode().is_unchecked_test_only()
}

pub fn with_raster_integrity_mode<T>(mode: RasterIntegrityMode, f: impl FnOnce() -> T) -> T {
    RASTER_INTEGRITY_MODE.with(|cell| {
        let previous = cell.replace(mode);
        let _reset = ResetRasterIntegrityMode { cell, previous };
        f()
    })
}

struct ResetRasterIntegrityMode<'a> {
    cell: &'a Cell<RasterIntegrityMode>,
    previous: RasterIntegrityMode,
}

impl Drop for ResetRasterIntegrityMode<'_> {
    fn drop(&mut self) {
        self.cell.set(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_mode_restores_previous_value() {
        assert_eq!(
            current_raster_integrity_mode(),
            RasterIntegrityMode::Verified
        );

        with_raster_integrity_mode(RasterIntegrityMode::Verified, || {
            assert_eq!(
                current_raster_integrity_mode(),
                RasterIntegrityMode::Verified
            );
        });

        assert_eq!(
            current_raster_integrity_mode(),
            RasterIntegrityMode::Verified
        );
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn unchecked_mode_is_scoped() {
        with_raster_integrity_mode(RasterIntegrityMode::UncheckedTestOnly, || {
            assert!(raster_integrity_is_unchecked());
        });

        assert!(!raster_integrity_is_unchecked());
    }
}
