#[macro_export]
macro_rules! new_t {
    ($t:ty, $($k:ident = $v:expr),+ $(,)?) => {{
        let mut c = <$t>::default();
        $(c.$k = $v;)+
        c
    }};
    ($($k:ident = $v:expr),+ $(,)?) => {new_t!(T, $($k = $v,)+)};
}

#[macro_export]
macro_rules! test_t {
  ( #[should_panic(expected = $panic_msg:literal)]
    $name:ident[T: $($impl:ty)|*]() $body:tt ) => {$(
      ::paste::paste! {
          #[test]
          #[should_panic(expected = $panic_msg)]
          fn [<$name _ $impl:snake>]() {
            type T=$impl;
            $body
          }
      }
  )+};
  ($name:ident[T: $($impl:ty)|*]() $body:tt ) => {$(
      ::paste::paste! {
          #[test]
          fn [<test_ $name _ $impl:snake>]() {
            type T=$impl;
            $body
          }
      }
  )+};
}

#[macro_export]
macro_rules! assert_f64_approx {
    ($l:expr, $r:expr) => {
        assert!(
            ($l - $r).abs() < f64::EPSILON,
            "assertion failed: {} !~ {}",
            $l,
            $r
        )
    };
    ($l:expr, $r:expr, $msg:expr) => {
        assert!(
            ($l - $r).abs() < f64::EPSILON,
            "assertion failed: {} !~ {}: {}",
            $l,
            $r,
            $msg
        )
    };
}

#[macro_export]
macro_rules! assert_matrix_approx {
    ($a:expr, $b:expr) => {
        assert_eq!($a.len(), $b.len(), "Matrices have different lengths");

        for (i, (l, r)) in $a.iter().zip($b.iter()).enumerate() {
            $crate::assert_f64_approx!(l, r, format!("differs at [{i}]"));
        }
    };
}

#[macro_export]
macro_rules! normalized {
    ($x:expr; $({.$($norm:tt)+})+) => {{
        let mut x = $x;
        $(x.$($norm)*;)+
        x
    }};
}

#[macro_export]
macro_rules! assert_some_normalized {
  ($l:expr, [$($r:expr),*  $(,)?]; $({.$($norm:tt)+})+, $msg: expr) => {{
    let l = $crate::normalized!($l.to_owned(); $({.$($norm)* })+);
    assert!([$($r,)*].into_iter().any(|r| l == $crate::normalized!(r.to_owned(); $({.$($norm)* })+)), "{}", $msg)
  }};
  ($l:expr, [$($r:expr),* $(,)?]; $({.$($norm:tt)+})+) => {$crate::assert_some_normalized!($l, [$($r,)*]; $({.$($norm)* })+, format!("{:?} not in {:?}", $l, [$($r,)*]))};
}
