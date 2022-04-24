pub use blis_sys as sys;
pub use reborrow::Reborrow;
pub use reborrow::ReborrowMut;

use core::marker::PhantomData;
use core::ptr::NonNull;

struct Inner<T> {
    buf: NonNull<T>,
    nrows: usize,
    ncols: usize,
    rs: isize,
    cs: isize,
}

impl<T> Clone for Inner<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Inner<T> {}

/// Immutable strided view over a matrix.
pub struct MatrixRef<'a, T> {
    inner: Inner<T>,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T> Clone for MatrixRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for MatrixRef<'a, T> {}

/// Mutable strided view over a matrix.
pub struct MatrixMut<'a, T> {
    inner: Inner<T>,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, 'b, T> Reborrow<'b> for MatrixMut<'a, T> {
    type Target = MatrixRef<'b, T>;

    fn rb(&'b self) -> Self::Target {
        Self::Target {
            inner: self.inner,
            _marker: PhantomData,
        }
    }
}

impl<'a, 'b, T> ReborrowMut<'b> for MatrixMut<'a, T> {
    type Target = MatrixMut<'b, T>;

    fn rb_mut(&'b mut self) -> Self::Target {
        Self::Target {
            inner: self.inner,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> MatrixRef<'a, T> {
    /// Returns a view from a pointer to the first element,
    /// the matrix dimensions and its strides.
    pub unsafe fn from_raw_parts(
        buf: NonNull<T>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: Inner {
                buf,
                nrows,
                ncols,
                rs: row_stride,
                cs: col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a view with 0 rows.
    pub fn new_0xn(ncols: usize) -> Self {
        unsafe { Self::from_raw_parts(NonNull::<T>::dangling(), 0, ncols, 0, 0) }
    }

    /// Returns a view with 0 columns.
    pub fn new_mx0(nrows: usize) -> Self {
        unsafe { Self::from_raw_parts(NonNull::<T>::dangling(), nrows, 0, 0, 0) }
    }

    /// Returns a view with 0 rows and 0 columns.
    pub fn new_0x0() -> Self {
        Self::new_mx0(0)
    }

    /// Returns a view that refers to a single element.
    pub fn new_1x1(value: &'a T) -> Self {
        let value = value as *const T as *mut T;
        unsafe { Self::from_raw_parts(NonNull::<T>::new_unchecked(value), 1, 1, 0, 0) }
    }

    /// Returns a view from a mutable slice, the matrix dimensions and its strides.
    pub fn try_from_slice(
        buf: &'a [T],
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self, DimsErrorMut> {
        let rs = row_stride;
        let cs = col_stride;
        unsafe {
            if nrows == 0 || ncols == 0 {
                Ok(Self::from_raw_parts(
                    NonNull::new_unchecked(buf.as_ptr() as *mut _),
                    nrows,
                    ncols,
                    0,
                    0,
                ))
            } else {
                let offset = largest_offset(nrows, ncols, rs, cs)?;
                if offset == usize::MAX {
                    return Err(DimsError::SizeOverflow.into());
                }
                if offset >= buf.len() {
                    return Err(DimsError::BufferTooSmall(offset + 1).into());
                }

                Ok(Self::from_raw_parts(
                    NonNull::new_unchecked(buf.as_ptr() as *mut _),
                    nrows,
                    ncols,
                    rs as isize,
                    cs as isize,
                ))
            }
        }
    }

    /// Returns a view over the transpose of `self`.
    pub fn trans(self) -> Self {
        unsafe {
            Self::from_raw_parts(
                self.inner.buf,
                self.inner.ncols,
                self.inner.nrows,
                self.inner.cs,
                self.inner.rs,
            )
        }
    }

    /// Returns the number of rows of the matrix.
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }

    /// Returns the number of columns of the matrix.
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Returns the stride between consecutive rows in the matrix.
    pub fn row_stride(&self) -> isize {
        self.inner.rs
    }

    /// Returns the stride between consecutive columns in the matrix.
    pub fn col_stride(&self) -> isize {
        self.inner.cs
    }
}

impl<'a, T> MatrixMut<'a, T> {
    /// Returns a mutable view from a pointer to the first element,
    /// the matrix dimensions and its strides.
    pub unsafe fn from_raw_parts_mut(
        buf: NonNull<T>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: Inner {
                buf,
                nrows,
                ncols,
                rs: row_stride,
                cs: col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a view with 0 rows.
    pub fn new_0xn(ncols: usize) -> Self {
        unsafe { Self::from_raw_parts_mut(NonNull::<T>::dangling(), 0, ncols, 0, 0) }
    }

    /// Returns a view with 0 columns.
    pub fn new_mx0(nrows: usize) -> Self {
        unsafe { Self::from_raw_parts_mut(NonNull::<T>::dangling(), nrows, 0, 0, 0) }
    }

    /// Returns a view with 0 rows and 0 columns.
    pub fn new_0x0() -> Self {
        Self::new_mx0(0)
    }

    /// Returns a mutable view that refers to a single element.
    pub fn new_1x1(value: &'a mut T) -> Self {
        unsafe { Self::from_raw_parts_mut(NonNull::<T>::new_unchecked(value), 1, 1, 0, 0) }
    }

    /// Returns a mutable view from a mutable slice, the matrix dimensions and its strides.
    pub fn try_from_mut_slice(
        buf: &'a mut [T],
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self, DimsErrorMut> {
        let rs = row_stride;
        let cs = col_stride;
        unsafe {
            if nrows == 0 || ncols == 0 {
                Ok(Self::from_raw_parts_mut(
                    NonNull::new_unchecked(buf.as_mut_ptr()),
                    nrows,
                    ncols,
                    0,
                    0,
                ))
            } else {
                let offset = largest_offset(nrows, ncols, rs, cs)?;
                if offset == usize::MAX {
                    return Err(DimsError::SizeOverflow.into());
                }
                if offset >= buf.len() {
                    return Err(DimsError::BufferTooSmall(offset + 1).into());
                }

                // i0, i1 in [0, m)
                // j0, j1 in [0, n)
                // s.t.
                // rs * i0 + cs * j0 == rs * i1 + cs * j1;
                //
                // => rs * (i1 - i0) + cs * (j1 - j0) == 0
                //
                // it's sufficient to find a solution of the form (i, 0), (0, j)
                //
                // i in [0, m)
                // j in [0, n)
                // rs * i = cs * j
                //
                //
                // let s = lcm(rs, cs)
                // let r = gcd(rs, cs)
                //
                // i = s / rs = cs / r
                // j = s / cs = rs / r
                //
                // check that they're within bounds
                if rs == 0 && nrows > 1 {
                    return Err(DimsErrorMut::SelfAlias(1, 0));
                }
                if cs == 0 && ncols > 1 {
                    return Err(DimsErrorMut::SelfAlias(0, 1));
                }
                if rs == 0 || cs == 0 {
                    return Ok(Self::from_raw_parts_mut(
                        NonNull::new_unchecked(buf.as_mut_ptr()),
                        nrows,
                        ncols,
                        rs as isize,
                        cs as isize,
                    ));
                }

                let r = gcd(rs, cs);

                let (i, j) = (cs / r, rs / r);
                if i < nrows && j < ncols {
                    return Err(DimsErrorMut::SelfAlias(i, j));
                }

                Ok(Self::from_raw_parts_mut(
                    NonNull::new_unchecked(buf.as_mut_ptr()),
                    nrows,
                    ncols,
                    rs as isize,
                    cs as isize,
                ))
            }
        }
    }

    /// Returns a mutable view over the transpose of `self`.
    pub fn trans(self) -> Self {
        unsafe {
            Self::from_raw_parts_mut(
                self.inner.buf,
                self.inner.ncols,
                self.inner.nrows,
                self.inner.cs,
                self.inner.rs,
            )
        }
    }

    /// Returns the number of rows of the matrix.
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }

    /// Returns the number of columns of the matrix.
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Returns the stride between consecutive rows in the matrix.
    pub fn row_stride(&self) -> isize {
        self.inner.rs
    }

    /// Returns the stride between consecutive columns in the matrix.
    pub fn col_stride(&self) -> isize {
        self.inner.cs
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let tmp = a % b;
        a = b;
        b = tmp;
    }
    a
}

fn largest_offset(nrows: usize, ncols: usize, rs: usize, cs: usize) -> Result<usize, DimsError> {
    // element with highest address is at (nrows-1, ncols-1)
    // we need to check that
    // rs*(m-1) + cs*(n-1) < buf.len()
    // while accounting for overflow
    let offset0 = rs.checked_mul(nrows - 1);
    let offset1 = cs.checked_mul(ncols - 1);

    let (offset0, offset1) = match (offset0, offset1) {
        (Some(offset0), Some(offset1)) => (offset0, offset1),
        _ => return Err(DimsError::SizeOverflow.into()),
    };

    Ok(offset0
        .checked_add(offset1)
        .ok_or(DimsError::SizeOverflow)?)
}

macro_rules! ty_to_dt {
    (f64) => {
        sys::num_t_BLIS_DOUBLE
    };
    (f32) => {
        sys::num_t_BLIS_FLOAT
    };
}

macro_rules! matrix_to_obj {
    ($name: ident, $input: expr, $dt: tt) => {
        let _input = $input;
        let mut $name = ::core::mem::MaybeUninit::<$crate::sys::obj_t>::uninit();
        let $name = $name.as_mut_ptr();
        unsafe {
            sys::bli_obj_create_without_buffer(
                ty_to_dt!($dt),
                _input.inner.nrows.try_into().unwrap(),
                _input.inner.ncols.try_into().unwrap(),
                $name,
            );
            (*$name).rs = _input.inner.rs.try_into().unwrap();
            (*$name).cs = _input.inner.cs.try_into().unwrap();
            (*$name).buffer = _input.inner.buf.as_ptr() as *mut ::std::os::raw::c_void;
        }
    };
}

macro_rules! set_uplo {
    ($obj: expr, $uplo: expr) => {
        unsafe {
            let obj: *mut $crate::sys::obj_t = $obj;
            let info = &mut *::core::ptr::addr_of_mut!((*obj).info);
            *info = (*info & !$crate::sys::BLIS_UPLO_BITS) | $uplo;
        }
    };
}

macro_rules! set_struc {
    ($obj: expr, $struc: expr) => {
        unsafe {
            let obj: *mut $crate::sys::obj_t = $obj;
            let info = &mut *::core::ptr::addr_of_mut!((*obj).info);
            *info = (*info & !$crate::sys::BLIS_STRUC_BITS) | $struc;
        }
    };
}

macro_rules! set_diag {
    ($obj: expr, $diag: expr) => {
        unsafe {
            let obj: *mut $crate::sys::obj_t = $obj;
            let info = &mut *::core::ptr::addr_of_mut!((*obj).info);
            *info = (*info & !$crate::sys::BLIS_UNIT_DIAG_BIT) | $diag;
        }
    };
}

pub trait Blis: Sized {
    /// If `beta == 0.0`, performs the operation
    /// `dst := alpha * lhs * rhs`.  
    /// Otherwise, performs the operation
    /// `dst := beta * dst + alpha * lhs * rhs`.
    fn gemm<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        beta: Self,
        num_threads: usize,
    );

    /// This operation is similar to `gemm`, but only the lower triangular part is updated.
    ///
    /// Let `actual_dst` be the lower triangular part of `dst`.  
    /// If `beta == 0.0`, performs the operation
    /// `actual_dst := alpha * lhs * rhs`.  
    /// Otherwise, performs the operation
    /// `actual_dst := beta * actual_dst + alpha * lhs * rhs`.
    fn gemmt<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        beta: Self,
        num_threads: usize,
    );

    /// Performs a triangular matrix product accumulate operation.
    ///
    /// Let `actual_lhs` be the lower triangular part of `lhs`.  
    /// If `beta == 0.0`, performs the operation  
    /// `dst := alpha * actual_lhs * rhs`.  
    /// Otherwise, performs the operation
    /// `dst := beta * dst + alpha * actual_lhs * rhs`.
    fn trmm3_left<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        beta: Self,
        num_threads: usize,
    );

    /// Performs a triangular matrix product accumulate operation.
    ///
    /// Let `actual_rhs` be the lower triangular part of `rhs`.  
    /// If `beta == 0.0`, performs the operation  
    /// `dst := alpha * lhs * actual_rhs`.  
    /// Otherwise, performs the operation
    /// `dst := beta * dst + alpha * lhs * actual_rhs`.
    fn trmm3_right<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        beta: Self,
        num_threads: usize,
    );

    /// Performs a triangular matrix product accumulate operation, assuming unit diagonal.
    ///
    /// Let `actual_lhs` be the lower triangular part of `lhs`, with an implicit unit diagonal.  
    /// If `beta == 0.0`, performs the operation  
    /// `dst := alpha * actual_lhs * rhs`.  
    /// Otherwise, performs the operation
    /// `dst := beta * dst + alpha * actual_lhs * rhs`.
    fn trmm3_left_unit_diag<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        beta: Self,
        num_threads: usize,
    );

    /// Performs a triangular matrix product accumulate operation.
    ///
    /// Let `actual_rhs` be the lower triangular part of `rhs`, with an implicit unit diagonal.  
    /// If `beta == 0.0`, performs the operation  
    /// `dst := alpha * lhs * actual_rhs`.  
    /// Otherwise, performs the operation
    /// `dst := beta * dst + alpha * lhs * actual_rhs`.
    fn trmm3_right_unit_diag<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        beta: Self,
        num_threads: usize,
    );

    /// Performs an in-place triangular matrix product operation.
    ///
    /// Let `actual_lhs` be the lower triangular part of `lhs`.  
    /// Performs the operation
    /// `dst := alpha * actual_lhs * dst`.
    fn trmm_left<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );

    /// Performs an in-place triangular matrix product operation.
    ///
    /// Let `actual_rhs` be the lower triangular part of `rhs`.  
    /// Performs the operation  
    /// `dst := alpha * dst * actual_rhs`.  
    fn trmm_right<'a>(
        dst: MatrixMut<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );

    /// Performs an in-place triangular matrix product operation, with unit diagonal.
    ///
    /// Let `actual_lhs` be the lower triangular part of `lhs`, with an implicit unit diagonal.  
    /// Performs the operation
    /// `dst := alpha * actual_lhs * dst`.
    fn trmm_left_unit_diag<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );

    /// Performs an in-place triangular matrix product operation, with unit diagonal.
    ///
    /// Let `actual_rhs` be the lower triangular part of `rhs`, with an implicit unit diagonal.  
    /// Performs the operation  
    /// `dst := alpha * dst * actual_rhs`.  
    fn trmm_right_unit_diag<'a>(
        dst: MatrixMut<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );

    /// Solves a triangular system in-place.
    ///
    /// Let `actual_lhs` be the lower triangular part of `lhs`.  
    /// Solves the equation
    /// `actual_lhs * X := alpha * dst`,  
    /// and stores the result in `dst`.
    fn trsm_left<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );

    /// Solves a triangular system in-place.
    ///
    /// Let `actual_rhs` be the lower triangular part of `rhs`.  
    /// Solves the equation
    /// `X * actual_rhs := alpha * dst`,  
    /// and stores the result in `dst`.
    fn trsm_right<'a>(
        dst: MatrixMut<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );

    /// Solves a triangular system in-place, with unit diagonal.
    ///
    /// Let `actual_lhs` be the lower triangular part of `lhs`, with an implicit unit diagonal.  
    /// Solves the equation
    /// `actual_lhs * X := alpha * dst`,  
    /// and stores the result in `dst`.
    fn trsm_left_unit_diag<'a>(
        dst: MatrixMut<'a, Self>,
        lhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );

    /// Solves a triangular system in-place, with unit diagonal.
    ///
    /// Let `actual_rhs` be the lower triangular part of `rhs`, with an implicit unit diagonal.  
    /// Solves the equation
    /// `X * actual_rhs := alpha * dst`,  
    /// and stores the result in `dst`.
    fn trsm_right_unit_diag<'a>(
        dst: MatrixMut<'a, Self>,
        rhs: MatrixRef<'a, Self>,
        alpha: Self,
        num_threads: usize,
    );
}

macro_rules! impl_blis {
    ($ty: tt) => {
        impl Blis for $ty {
            fn gemm<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                beta: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);
                let beta = MatrixRef::new_1x1(&beta);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);
                matrix_to_obj!(obj_beta, beta, $ty);

                unsafe {
                    sys::bli_gemm_ex(
                        obj_alpha,
                        obj_lhs,
                        obj_rhs,
                        obj_beta,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn gemmt<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                beta: Self,
                num_threads: usize,
            ) {
                assert_eq!(
                    dst.inner.nrows, dst.inner.ncols,
                    "Destination must be a square matrix"
                );
                let alpha = MatrixRef::new_1x1(&alpha);
                let beta = MatrixRef::new_1x1(&beta);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);
                matrix_to_obj!(obj_beta, beta, $ty);

                set_uplo!(obj_dst, sys::uplo_t_BLIS_LOWER);

                unsafe {
                    sys::bli_gemmt_ex(
                        obj_alpha,
                        obj_lhs,
                        obj_rhs,
                        obj_beta,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm3_left<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                beta: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);
                let beta = MatrixRef::new_1x1(&beta);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);
                matrix_to_obj!(obj_beta, beta, $ty);

                set_uplo!(obj_lhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_lhs, sys::diag_t_BLIS_NONUNIT_DIAG);
                set_struc!(obj_lhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm3_ex(
                        sys::side_t_BLIS_LEFT,
                        obj_alpha,
                        obj_lhs,
                        obj_rhs,
                        obj_beta,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm3_right<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                beta: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);
                let beta = MatrixRef::new_1x1(&beta);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);
                matrix_to_obj!(obj_beta, beta, $ty);

                set_uplo!(obj_rhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_rhs, sys::diag_t_BLIS_NONUNIT_DIAG);
                set_struc!(obj_rhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm3_ex(
                        sys::side_t_BLIS_RIGHT,
                        obj_alpha,
                        obj_rhs,
                        obj_lhs,
                        obj_beta,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm3_left_unit_diag<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                beta: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);
                let beta = MatrixRef::new_1x1(&beta);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);
                matrix_to_obj!(obj_beta, beta, $ty);

                set_uplo!(obj_lhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_lhs, sys::diag_t_BLIS_UNIT_DIAG);
                set_struc!(obj_lhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm3_ex(
                        sys::side_t_BLIS_LEFT,
                        obj_alpha,
                        obj_lhs,
                        obj_rhs,
                        obj_beta,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm3_right_unit_diag<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                beta: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);
                let beta = MatrixRef::new_1x1(&beta);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);
                matrix_to_obj!(obj_beta, beta, $ty);

                set_uplo!(obj_rhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_rhs, sys::diag_t_BLIS_UNIT_DIAG);
                set_struc!(obj_rhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm3_ex(
                        sys::side_t_BLIS_RIGHT,
                        obj_alpha,
                        obj_rhs,
                        obj_lhs,
                        obj_beta,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm_left<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_lhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_lhs, sys::diag_t_BLIS_NONUNIT_DIAG);
                set_struc!(obj_lhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm_ex(
                        sys::side_t_BLIS_LEFT,
                        obj_alpha,
                        obj_lhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm_right<'a>(
                dst: MatrixMut<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_rhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_rhs, sys::diag_t_BLIS_NONUNIT_DIAG);
                set_struc!(obj_rhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm_ex(
                        sys::side_t_BLIS_RIGHT,
                        obj_alpha,
                        obj_rhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm_left_unit_diag<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_lhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_lhs, sys::diag_t_BLIS_UNIT_DIAG);
                set_struc!(obj_lhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm_ex(
                        sys::side_t_BLIS_LEFT,
                        obj_alpha,
                        obj_lhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trmm_right_unit_diag<'a>(
                dst: MatrixMut<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_rhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_rhs, sys::diag_t_BLIS_UNIT_DIAG);
                set_struc!(obj_rhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trmm_ex(
                        sys::side_t_BLIS_RIGHT,
                        obj_alpha,
                        obj_rhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trsm_left<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_lhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_lhs, sys::diag_t_BLIS_NONUNIT_DIAG);
                set_struc!(obj_lhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trsm_ex(
                        sys::side_t_BLIS_LEFT,
                        obj_alpha,
                        obj_lhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trsm_right<'a>(
                dst: MatrixMut<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_rhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_rhs, sys::diag_t_BLIS_NONUNIT_DIAG);
                set_struc!(obj_rhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trsm_ex(
                        sys::side_t_BLIS_RIGHT,
                        obj_alpha,
                        obj_rhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trsm_left_unit_diag<'a>(
                dst: MatrixMut<'a, Self>,
                lhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_lhs, lhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_lhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_lhs, sys::diag_t_BLIS_UNIT_DIAG);
                set_struc!(obj_lhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trsm_ex(
                        sys::side_t_BLIS_LEFT,
                        obj_alpha,
                        obj_lhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }

            fn trsm_right_unit_diag<'a>(
                dst: MatrixMut<'a, Self>,
                rhs: MatrixRef<'a, Self>,
                alpha: Self,
                num_threads: usize,
            ) {
                let alpha = MatrixRef::new_1x1(&alpha);

                matrix_to_obj!(obj_dst, dst, $ty);
                matrix_to_obj!(obj_rhs, rhs, $ty);
                matrix_to_obj!(obj_alpha, alpha, $ty);

                set_uplo!(obj_rhs, sys::uplo_t_BLIS_LOWER);
                set_diag!(obj_rhs, sys::diag_t_BLIS_UNIT_DIAG);
                set_struc!(obj_rhs, sys::struc_t_BLIS_TRIANGULAR);

                unsafe {
                    sys::bli_trsm_ex(
                        sys::side_t_BLIS_RIGHT,
                        obj_alpha,
                        obj_rhs,
                        obj_dst,
                        core::ptr::null(),
                        &mut to_rntm(num_threads),
                    );
                }
            }
        }
    };
}

impl_blis!(f64);
impl_blis!(f32);

/// Error during the construction of a matrix view.
#[derive(Copy, Clone, Debug)]
pub enum DimsError {
    /// The size of the matrix does not fit in a `usize`.
    SizeOverflow,
    /// The buffer is too small. This variant contains the required buffer size.
    BufferTooSmall(usize),
}

/// Error during the construction of a mutable matrix view.
#[derive(Copy, Clone, Debug)]
pub enum DimsErrorMut {
    /// Dimension error.
    DimsError(DimsError),
    /// The address at some (i, j) aliases the address at (0, 0).
    /// This variant contains the indices `i` and `j`.
    SelfAlias(usize, usize),
}

impl From<DimsError> for DimsErrorMut {
    fn from(e: DimsError) -> Self {
        Self::DimsError(e)
    }
}

#[inline]
fn to_rntm(num_threads: usize) -> sys::rntm_t {
    sys::rntm_t {
        auto_factor: true,
        num_threads: num_threads.try_into().unwrap(),
        thrloop: [-1, -1, -1, -1, -1, -1],
        pack_a: false,
        pack_b: false,
        l3_sup: true,
        sba_pool: core::ptr::null_mut(),
        pba: core::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_buf() {
        let mut c = [0.0, 0.0];
        assert!(MatrixMut::try_from_mut_slice(&mut c, 3, 1, 1, 0).is_err());
        assert!(MatrixRef::try_from_slice(&c, 3, 1, 1, 0).is_err());
    }

    #[test]
    fn no_self_alias() {
        let mut c = [0.0, 0.0];
        assert!(MatrixMut::try_from_mut_slice(&mut c, 2, 1, 1, 0).is_ok());
        assert!(MatrixMut::try_from_mut_slice(&mut c, 1, 2, 0, 1).is_ok());
        assert!(MatrixMut::try_from_mut_slice(&mut c, 1, 1, 0, 0).is_ok());
        assert!(MatrixMut::try_from_mut_slice(&mut c, 2, 1, 0, 0).is_err());
        assert!(MatrixMut::try_from_mut_slice(&mut c, 1, 2, 0, 0).is_err());
    }

    #[test]
    fn self_alias() {
        let c = [0.0, 0.0];
        assert!(MatrixRef::try_from_slice(&c, 2, 1, 1, 0).is_ok());
        assert!(MatrixRef::try_from_slice(&c, 1, 2, 0, 1).is_ok());
        assert!(MatrixRef::try_from_slice(&c, 1, 1, 0, 0).is_ok());
        assert!(MatrixRef::try_from_slice(&c, 2, 1, 0, 0).is_ok());
        assert!(MatrixRef::try_from_slice(&c, 1, 2, 0, 0).is_ok());
    }

    macro_rules! impl_gemm_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;
                let beta = 2.0;

                let a = [1.0, 2.0, 3.0, 4.0];
                let b = [5.0, 6.0];
                let c = [10.0, 20.0];
                let mut c0 = c;
                let mut c1 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let b = MatrixRef::try_from_slice(&b, 2, 1, 1, 0).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 1, 1, 0).unwrap();
                    let c1 = MatrixMut::try_from_mut_slice(&mut c1, 2, 1, 1, 0).unwrap();

                    <$ty>::gemm(c0, a, b, alpha, beta, 1);
                    <$ty>::gemm(c1, a.trans(), b, alpha, beta, 1);
                }
                assert_eq!(
                    c0,
                    [
                        beta * c[0] + alpha * (a[0] * b[0] + a[2] * b[1]),
                        beta * c[1] + alpha * (a[1] * b[0] + a[3] * b[1]),
                    ]
                );
                assert_eq!(
                    c1,
                    [
                        beta * c[0] + alpha * (a[0] * b[0] + a[1] * b[1]),
                        beta * c[1] + alpha * (a[2] * b[0] + a[3] * b[1]),
                    ]
                );
            }
        };
    }

    impl_gemm_test!(gemm_f32, f32);
    impl_gemm_test!(gemm_f64, f64);

    macro_rules! impl_gemmt_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;
                let beta = 2.0;

                let a = [1.0, 2.0, 3.0, 4.0];
                let b = [5.0, 6.0, 7.0, 8.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let b = MatrixRef::try_from_slice(&b, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::gemmt(c0, a, b, alpha, beta, 1);
                }
                assert_eq!(
                    c0,
                    [
                        beta * c[0] + alpha * (a[0] * b[0] + a[2] * b[1]),
                        beta * c[1] + alpha * (a[1] * b[0] + a[3] * b[1]),
                        c[2],
                        beta * c[3] + alpha * (a[1] * b[2] + a[3] * b[3]),
                    ]
                );
            }
        };
    }

    impl_gemmt_test!(gemmt_f32, f32);
    impl_gemmt_test!(gemmt_f64, f64);

    macro_rules! impl_trmm3_left_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;
                let beta = 2.0;

                let a = [1.0, 2.0, 3.0, 4.0];
                let b = [5.0, 6.0, 7.0, 8.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let b = MatrixRef::try_from_slice(&b, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm3_left(c0, a, b, alpha, beta, 1);
                }
                assert_eq!(
                    c0,
                    [
                        beta * c[0] + alpha * (a[0] * b[0]),
                        beta * c[1] + alpha * (a[1] * b[0] + a[3] * b[1]),
                        beta * c[2] + alpha * (a[0] * b[2]),
                        beta * c[3] + alpha * (a[1] * b[2] + a[3] * b[3]),
                    ]
                );
            }
        };
    }

    macro_rules! impl_trmm3_left_unit_diag_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;
                let beta = 2.0;

                let a = [1.0, 2.0, 3.0, 4.0];
                let b = [5.0, 6.0, 7.0, 8.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let b = MatrixRef::try_from_slice(&b, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm3_left_unit_diag(c0, a, b, alpha, beta, 1);
                }
                assert_eq!(
                    c0,
                    [
                        beta * c[0] + alpha * (1.0 * b[0]),
                        beta * c[1] + alpha * (a[1] * b[0] + 1.0 * b[1]),
                        beta * c[2] + alpha * (1.0 * b[2]),
                        beta * c[3] + alpha * (a[1] * b[2] + 1.0 * b[3]),
                    ]
                );
            }
        };
    }

    macro_rules! impl_trmm3_right_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;
                let beta = 2.0;

                let a = [1.0, 2.0, 3.0, 4.0];
                let b = [5.0, 6.0, 7.0, 8.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let b = MatrixRef::try_from_slice(&b, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm3_right(c0, a, b, alpha, beta, 1);
                }
                assert_eq!(
                    c0,
                    [
                        beta * c[0] + alpha * (a[0] * b[0] + a[2] * b[1]),
                        beta * c[1] + alpha * (a[1] * b[0] + a[3] * b[1]),
                        beta * c[2] + alpha * (a[2] * b[3]),
                        beta * c[3] + alpha * (a[3] * b[3]),
                    ]
                );
            }
        };
    }

    macro_rules! impl_trmm3_right_unit_diag_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;
                let beta = 2.0;

                let a = [1.0, 2.0, 3.0, 4.0];
                let b = [5.0, 6.0, 7.0, 8.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let b = MatrixRef::try_from_slice(&b, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm3_right_unit_diag(c0, a, b, alpha, beta, 1);
                }
                assert_eq!(
                    c0,
                    [
                        beta * c[0] + alpha * (a[0] * 1.0 + a[2] * b[1]),
                        beta * c[1] + alpha * (a[1] * 1.0 + a[3] * b[1]),
                        beta * c[2] + alpha * (a[2] * 1.0),
                        beta * c[3] + alpha * (a[3] * 1.0),
                    ]
                );
            }
        };
    }

    impl_trmm3_left_test!(trmm3l_f32, f32);
    impl_trmm3_left_test!(trmm3l_f64, f64);
    impl_trmm3_left_unit_diag_test!(trmm3lu_f32, f32);
    impl_trmm3_left_unit_diag_test!(trmm3lu_f64, f64);
    impl_trmm3_right_test!(trmm3r_f32, f32);
    impl_trmm3_right_test!(trmm3r_f64, f64);
    impl_trmm3_right_unit_diag_test!(trmm3ru_f32, f32);
    impl_trmm3_right_unit_diag_test!(trmm3ru_f64, f64);

    macro_rules! impl_trmm_left_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;

                let a = [1.0, 2.0, 3.0, 4.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm_left(c0, a, alpha, 1);
                }
                assert_eq!(
                    c0,
                    [
                        alpha * (a[0] * c[0]),
                        alpha * (a[1] * c[0] + a[3] * c[1]),
                        alpha * (a[0] * c[2]),
                        alpha * (a[1] * c[2] + a[3] * c[3]),
                    ]
                );
            }
        };
    }

    macro_rules! impl_trmm_left_unit_diag_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;

                let a = [1.0, 2.0, 3.0, 4.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let a = MatrixRef::try_from_slice(&a, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm_left_unit_diag(c0, a, alpha, 1);
                }
                assert_eq!(
                    c0,
                    [
                        alpha * (1.0 * c[0]),
                        alpha * (a[1] * c[0] + 1.0 * c[1]),
                        alpha * (1.0 * c[2]),
                        alpha * (a[1] * c[2] + 1.0 * c[3]),
                    ]
                );
            }
        };
    }

    macro_rules! impl_trmm_right_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;

                let b = [5.0, 6.0, 7.0, 8.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let b = MatrixRef::try_from_slice(&b, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm_right(c0, b, alpha, 1);
                }
                assert_eq!(
                    c0,
                    [
                        alpha * (c[0] * b[0] + c[2] * b[1]),
                        alpha * (c[1] * b[0] + c[3] * b[1]),
                        alpha * (c[2] * b[3]),
                        alpha * (c[3] * b[3]),
                    ]
                );
            }
        };
    }

    macro_rules! impl_trmm_right_unit_diag_test {
        ($name: ident, $ty: ty) => {
            #[test]
            fn $name() {
                let alpha = 0.5;

                let b = [5.0, 6.0, 7.0, 8.0];
                let c = [10.0, 20.0, 30.0, 40.0];
                let mut c0 = c;

                {
                    let b = MatrixRef::try_from_slice(&b, 2, 2, 1, 2).unwrap();
                    let c0 = MatrixMut::try_from_mut_slice(&mut c0, 2, 2, 1, 2).unwrap();

                    <$ty>::trmm_right_unit_diag(c0, b, alpha, 1);
                }
                assert_eq!(
                    c0,
                    [
                        alpha * (c[0] * 1.0 + c[2] * b[1]),
                        alpha * (c[1] * 1.0 + c[3] * b[1]),
                        alpha * (c[2] * 1.0),
                        alpha * (c[3] * 1.0),
                    ]
                );
            }
        };
    }

    impl_trmm_left_test!(trmml_f32, f32);
    impl_trmm_left_test!(trmml_f64, f64);
    impl_trmm_left_unit_diag_test!(trmmlu_f32, f32);
    impl_trmm_left_unit_diag_test!(trmmlu_f64, f64);
    impl_trmm_right_test!(trmmr_f32, f32);
    impl_trmm_right_test!(trmmr_f64, f64);
    impl_trmm_right_unit_diag_test!(trmmru_f32, f32);
    impl_trmm_right_unit_diag_test!(trmmru_f64, f64);

    // TODO
    // trsm tests
}
