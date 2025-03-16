use crate::{
    genome::Connection,
    random::{CHANCE_KEEP_DISABLED, CHANCE_NEW_DISABLED, CHANCE_PICK_L_EQ},
};
use core::cmp::Ordering;
use rand::{rngs::ThreadRng, Rng};

pub fn disjoint_excess_count(l: &[Connection], r: &[Connection]) -> (f64, f64) {
    let mut l_iter = l.iter();
    let mut r_iter = r.iter();

    let mut l_conn = match l_iter.next() {
        Some(c) => c,
        None => return (0., r_iter.count() as f64),
    };

    let mut r_conn = match r_iter.next() {
        Some(c) => c,
        None => return (0., l_iter.count() as f64 + 1.),
    };

    let mut disjoint = 0.;
    let excess_passed = loop {
        match l_conn.inno.cmp(&r_conn.inno) {
            Ordering::Equal => {
                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break 0.,
                };

                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break 1.,
                };
            }
            Ordering::Greater => {
                disjoint += 1.;
                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break 1.,
                }
            }
            Ordering::Less => {
                disjoint += 1.;
                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break 1.,
                }
            }
        }
    };

    (
        disjoint,
        l_iter.count() as f64 + r_iter.count() as f64 + excess_passed,
    )
}

/// if genomes share no overlapping weights, their average diff should be 0
pub fn avg_weight_diff(l: &[Connection], r: &[Connection]) -> f64 {
    let mut diff = 0.;
    let mut count = 0.;
    let mut l_iter = l.iter();
    let mut r_iter = r.iter();

    let mut l_conn = match l_iter.next() {
        Some(c) => c,
        None => return 0.,
    };

    let mut r_conn = match r_iter.next() {
        Some(c) => c,
        None => return 0.,
    };

    loop {
        match l_conn.inno.cmp(&r_conn.inno) {
            Ordering::Equal => {
                diff += (l_conn.weight - r_conn.weight).abs();
                count += 1.;

                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break,
                };

                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break,
                };
            }
            Ordering::Greater => {
                r_conn = match r_iter.next() {
                    Some(c) => c,
                    None => break,
                }
            }
            Ordering::Less => {
                l_conn = match l_iter.next() {
                    Some(c) => c,
                    None => break,
                }
            }
        }
    }

    if count == 0. {
        0.
    } else {
        diff / count
    }
}

const EXCESS_COEFFICIENT: f64 = 1.0;
const DISJOINT_COEFFICIENT: f64 = 1.0;
const WEIGHT_COEFFICIENT: f64 = 0.4;

pub fn delta(l: &[Connection], r: &[Connection]) -> f64 {
    let l_size = l.len() as f64;
    let r_size = r.len() as f64;
    let fac = {
        let longest = f64::max(l_size, r_size);
        if longest < 20. {
            1.
        } else {
            longest
        }
    };

    if l_size == 0. || r_size == 0. {
        (EXCESS_COEFFICIENT * f64::max(l_size, r_size)) / fac
    } else {
        let (disjoint, excess) = disjoint_excess_count(l, r);
        (DISJOINT_COEFFICIENT * disjoint + EXCESS_COEFFICIENT * excess) / fac
            + WEIGHT_COEFFICIENT * avg_weight_diff(l, r)
    }
}

#[inline]
fn pick_gene(
    base_conn: &Connection,
    opt_conn: Option<&Connection>,
    rng: &mut ThreadRng,
) -> Connection {
    let mut conn = if let Some(r_conn) = opt_conn {
        (*if rng.random_bool(CHANCE_PICK_L_EQ) {
            r_conn
        } else {
            base_conn
        })
        .to_owned()
    } else {
        base_conn.to_owned()
    };

    // TODO It seems like it will always check RAND_DISABLED, and sometimes
    // check KEEP_DISABLED. I wonder if checking RAND_DISABLED first would bypass
    // RAND_DISABLED% of checks that would then check KEEP_DISABLED?
    if ((!base_conn.enabled || opt_conn.is_some_and(|r_conn| !r_conn.enabled))
        && rng.random_bool(CHANCE_KEEP_DISABLED))
        || rng.random_bool(CHANCE_NEW_DISABLED)
    {
        conn.enabled = false;
    }

    conn
}

/// crossover connections where l and r are equally fit
fn crossover_eq(l: &[Connection], r: &[Connection], rng: &mut ThreadRng) -> Vec<Connection> {
    // TODO I wonder what the actual average case overlap between genomes is?
    // probably pretty close, could we measure this?
    let mut cross = Vec::with_capacity(l.len() + r.len());
    let mut l_idx = 0;
    let mut r_idx = 0;
    loop {
        match (l.get(l_idx), r.get(r_idx)) {
            (None, None) => break,
            (None, Some(_)) => {
                // TODO is it faster to extend, or to loop-push?
                cross.extend(r[r_idx..].iter().map(|conn| pick_gene(conn, None, rng)));
                break;
            }
            (Some(_), None) => {
                cross.extend(l[l_idx..].iter().map(|conn| pick_gene(conn, None, rng)));
                break;
            }
            (Some(l_conn), Some(r_conn)) => match l_conn.inno.cmp(&r_conn.inno) {
                Ordering::Equal => {
                    cross.push(pick_gene(l_conn, Some(r_conn), rng));
                    l_idx += 1;
                    r_idx += 1;
                }
                Ordering::Less => {
                    cross.push(pick_gene(l_conn, None, rng));
                    l_idx += 1;
                }
                Ordering::Greater => {
                    cross.push(pick_gene(r_conn, None, rng));
                    r_idx += 1;
                }
            },
        }
    }

    cross.shrink_to_fit(); // TODO what happens if I remove this
    cross
}

/// crossover connections where l is more fit than r
fn crossover_ne(l: &[Connection], r: &[Connection], rng: &mut ThreadRng) -> Vec<Connection> {
    // copy l, pick_gene where l.inno == r.inno
    let mut cross = Vec::with_capacity(l.len());
    let mut r_idx = 0;
    for l_conn in l {
        // TODO is r_idx < r.len() && r[r_idx] or maybe even get_unchecked
        while r.get(r_idx).is_some_and(|r_conn| r_conn.inno < l_conn.inno) {
            r_idx += 1;
        }

        // TODO above applies here
        cross.push(pick_gene(
            l_conn,
            r.get(r_idx)
                .is_some_and(|r_conn| r_conn.inno == l_conn.inno)
                .then(|| &r[r_idx]),
            rng,
        ))
    }

    cross
}

/// crossover connections
/// l_fit describes how fit l is compared to r,
pub fn crossover(
    l: &[Connection],
    r: &[Connection],
    l_fit: Ordering,
    rng: &mut ThreadRng,
) -> Vec<Connection> {
    let mut usort = match l_fit {
        Ordering::Equal => crossover_eq(l, r, rng),
        Ordering::Less => crossover_ne(r, l, rng),
        Ordering::Greater => crossover_ne(l, r, rng),
    };

    usort.sort_by_key(|c| c.inno);
    usort
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::genome::Connection;
    use rand::rng;
    use std::collections::{HashMap, HashSet};

    macro_rules! connection {
        ($($k:ident = $v:expr),+ $(,)?) => {{
            let mut c = Connection{
                inno: 0,
                from: 0,
                to: 0,
                weight: 0.,
                enabled: true,
            };
            $(c.$k = $v;)+
            c
          }}
    }

    #[test]
    fn test_avg_weight_diff() {
        let diff = avg_weight_diff(
            &[
                connection!(inno = 1, weight = 0.5,),
                connection!(inno = 2, weight = -0.5,),
                connection!(inno = 3, weight = 1.0,),
            ],
            &[
                connection!(inno = 1, weight = 0.0,),
                connection!(inno = 2, weight = -1.0,),
                connection!(inno = 4, weight = 2.0,),
            ],
        );
        assert!((diff - 0.5).abs() < f64::EPSILON, "diff ne: {diff}, 0.5");
    }

    #[test]
    fn test_avg_weight_diff_empty() {
        let full = vec![
            connection!(inno = 1, weight = 0.0,),
            connection!(inno = 2, weight = -1.0,),
            connection!(inno = 4, weight = 2.0,),
        ];

        let diff = avg_weight_diff(&full, &[]);
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");

        let diff = avg_weight_diff(&[], &full);
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");

        let diff = avg_weight_diff(&[], &[]);
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");
    }

    #[test]
    fn test_avg_weight_diff_no_overlap() {
        let diff = avg_weight_diff(
            &[
                connection!(inno = 1, weight = 0.5,),
                connection!(inno = 2, weight = -0.5,),
                connection!(inno = 3, weight = 1.0,),
            ],
            &[
                connection!(inno = 5, weight = 0.5,),
                connection!(inno = 6, weight = -0.5,),
            ],
        );
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");
    }

    #[test]
    fn test_avg_weight_diff_no_diff() {
        let diff = avg_weight_diff(
            &[
                connection!(inno = 1, weight = 0.5,),
                connection!(inno = 2, weight = -0.5,),
                connection!(inno = 3, weight = 1.0,),
            ],
            &[
                connection!(inno = 1, weight = 0.5,),
                connection!(inno = 2, weight = -0.5,),
                connection!(inno = 3, weight = 1.0,),
            ],
        );
        assert!((diff - 0.0).abs() < f64::EPSILON, "diff ne: {diff}, 0.");
    }

    #[test]
    fn test_disjoint_excess_count() {
        assert_eq!(
            (4.0, 2.0),
            disjoint_excess_count(
                &[
                    connection!(inno = 1),
                    connection!(inno = 2),
                    connection!(inno = 6),
                ],
                &[
                    connection!(inno = 1),
                    connection!(inno = 3),
                    connection!(inno = 4),
                    connection!(inno = 8),
                    connection!(inno = 10),
                ]
            )
        );
    }

    #[test]
    fn test_disjoint_excess_count_symmetrical() {
        let l = vec![
            connection!(inno = 1),
            connection!(inno = 2),
            connection!(inno = 6),
        ];
        let r = vec![
            connection!(inno = 1),
            connection!(inno = 3),
            connection!(inno = 4),
            connection!(inno = 8),
            connection!(inno = 10),
        ];
        assert_eq!(disjoint_excess_count(&l, &r), disjoint_excess_count(&r, &l));
    }

    #[test]
    fn test_disjoint_excess_count_empty() {
        let full = vec![connection!(inno = 1), connection!(inno = 2)];
        assert_eq!((0.0, 2.0), disjoint_excess_count(&full, &[]));
        assert_eq!((0.0, 2.0), disjoint_excess_count(&[], &full));
        assert_eq!((0.0, 0.0), disjoint_excess_count(&[], &[]));
    }

    #[test]
    fn test_disjoint_excess_count_hanging_l() {
        assert_eq!(
            (0.0, 1.0),
            disjoint_excess_count(
                &[
                    connection!(inno = 0),
                    connection!(inno = 1),
                    connection!(inno = 2),
                ],
                &[connection!(inno = 0), connection!(inno = 1),]
            )
        )
    }

    #[test]
    fn test_disjoint_excess_count_no_overlap() {
        assert_eq!(
            (2.0, 2.0),
            disjoint_excess_count(
                &[connection!(inno = 1), connection!(inno = 2),],
                &[connection!(inno = 3), connection!(inno = 4),]
            )
        );
    }

    #[test]
    fn test_disjoint_excess_count_short_larger_inno() {
        assert_eq!(
            (3.0, 1.0),
            disjoint_excess_count(
                &[connection!(inno = 10)],
                &[
                    connection!(inno = 1),
                    connection!(inno = 2),
                    connection!(inno = 3),
                ]
            )
        );
    }

    macro_rules! assert_from_connection {
        ($have:expr, ($l:expr, $r:expr), $($arg:tt)+) => {{
            let mut have = $have.to_owned();
            have.enabled = true;
            let mut l = $l.to_owned();
            l.enabled = true;
            let mut r = $r.to_owned();
            r.enabled = true;
            assert!(have == l || have == r, $($arg)*);
        }};
        ($have:expr, ($l:expr, $r:expr)) => {{
            assert_from_connection!(
                $have, ($l, $r),
                "{:?} from neither {:?} or {:?}",
                $have,
                $l,
                $r
            );
        }};
        ($have:expr, $f:expr, $($arg:tt)+) => {{
            let mut have = $have.to_owned();
            have.enabled = true;
            let mut f = $f.to_owned();
            f.enabled = true;
            assert!(have == f, $($arg)*);

        }};
        ($have:expr, $f:expr) => {{
            assert_from_connection!(
                $have, $f,
                "{:?} not from {:?}",
                $have,
                $f
            )
        }};
    }

    fn assert_crossover_eq(l: &[Connection], r: &[Connection]) {
        for (l, r) in [(l, r), (r, l)] {
            let l_map = l.iter().map(|c| (c.inno, c)).collect::<HashMap<_, &_>>();
            let r_map = r.iter().map(|c| (c.inno, c)).collect::<HashMap<_, &_>>();
            let inno = l_map
                .keys()
                .collect::<HashSet<_>>()
                .union(&r_map.keys().collect::<HashSet<_>>())
                .cloned()
                .cloned()
                .collect::<HashSet<_>>();

            for _ in 0..1000 {
                let lr = crossover_eq(l, r, &mut rng());
                assert_eq!(inno.len(), lr.len());

                let lr_inno = lr.iter().map(|c| c.inno).collect::<HashSet<_>>();
                assert!(inno.is_subset(&lr_inno));
                assert!(inno.is_superset(&lr_inno));
                assert!(lr.is_sorted_by_key(|c| c.inno));
                for ref lr_conn in lr {
                    match (l_map.get(&lr_conn.inno), r_map.get(&lr_conn.inno)) {
                        (None, None) => panic!("{} is in neither l nor r", lr_conn.inno),
                        (None, Some(conn)) | (Some(conn), None) => {
                            assert_from_connection!(lr_conn, *conn)
                        }
                        (Some(l_conn), Some(r_conn)) => {
                            assert_from_connection!(lr_conn, (*l_conn, *r_conn))
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_crossover_eq() {
        let l = [
            connection!(inno = 0, from = 1_1),
            connection!(inno = 1, from = 1_2),
            connection!(inno = 2, from = 1_3),
        ];
        let r = [
            connection!(inno = 0, from = 2_1),
            connection!(inno = 2, from = 2_2),
            connection!(inno = 3, from = 2_3),
        ];

        assert_crossover_eq(&l, &r);
    }

    #[test]
    fn test_crossover_eq_empty() {
        let l = [connection!(inno = 2, from = 1)];

        assert_crossover_eq(&l, &[]);
        assert_crossover_eq(&[], &[]);
    }

    #[test]
    fn test_crossover_eq_overflow() {
        let l = [connection!(inno = 0, from = 1_1)];
        let r = [connection!(inno = 1, from = 2_1)];

        assert_crossover_eq(&l, &r);

        let l = [connection!(inno = 1, from = 1_1)];
        let r = [connection!(inno = 0, from = 2_1)];

        assert_crossover_eq(&l, &r);
    }

    #[test]
    #[should_panic(expected = "not from r_0")]
    fn test_crossover_eq_catchup_l() {
        let l = [
            connection!(inno = 0, from = 1_1),
            connection!(inno = 1, from = 1_2),
        ];
        let r = [connection!(inno = 1, from = 2_1)];
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng());
            assert_eq!(lr.len(), 2);
            assert_from_connection!(lr[0], l[0]);
            assert_from_connection!(lr[1], r[0], "not from r_0");
        }
    }

    #[test]
    #[should_panic(expected = "not from l_0")]
    fn test_crossover_eq_catchup_r() {
        let l = [connection!(inno = 1, from = 2_1)];
        let r = [
            connection!(inno = 0, from = 1_1),
            connection!(inno = 1, from = 1_2),
        ];
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng());
            assert_eq!(lr.len(), 2);
            assert_from_connection!(lr[0], r[0]);
            assert_from_connection!(lr[1], l[0], "not from l_0");
        }
    }

    #[test]
    #[should_panic(expected = "not from l_1")]
    fn test_crossover_eq_both_step_l() {
        let l = [
            connection!(inno = 0, from = 1_1),
            connection!(inno = 1, from = 1_2),
        ];
        let r = [
            connection!(inno = 0, from = 2_1),
            connection!(inno = 1, from = 2_2),
        ];
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng());
            assert_eq!(lr.len(), 2);
            assert_from_connection!(lr[0], (l[0], r[0]));
            assert_from_connection!(lr[1], l[1], "not from l_1");
        }
    }

    #[test]
    #[should_panic(expected = "not from r_1")]
    fn test_crossover_eq_both_step_r() {
        let l = [
            connection!(inno = 0, from = 1_1),
            connection!(inno = 1, from = 1_2),
        ];
        let r = [
            connection!(inno = 0, from = 2_1),
            connection!(inno = 1, from = 2_2),
        ];
        for _ in 0..1000 {
            let lr = crossover_eq(&l, &r, &mut rng());
            assert_eq!(lr.len(), 2);
            assert_from_connection!(lr[0], (l[0], r[0]));
            assert_from_connection!(lr[1], r[1], "not from r_1");
        }
    }

    fn assert_crossover_ne(l: &[Connection], r: &[Connection]) {
        for (l, r) in [(l, r), (r, l)] {
            let l_map = l.iter().map(|c| (c.inno, c)).collect::<HashMap<_, &_>>();
            let r_map = r.iter().map(|c| (c.inno, c)).collect::<HashMap<_, &_>>();
            let l_keys = l_map.keys().cloned().collect::<HashSet<_>>();
            let inno = l_keys
                .union(&r_map.keys().cloned().collect::<HashSet<_>>())
                .cloned()
                .collect::<HashSet<_>>();

            for _ in 0..1000 {
                let lr = crossover_ne(l, r, &mut rng());
                assert_eq!(lr.len(), l.len());

                let lr_inno = lr.iter().map(|c| c.inno).collect::<HashSet<_>>();
                assert!(l_keys.is_subset(&lr_inno));
                assert!(l_keys.is_superset(&lr_inno));
                assert!(inno.is_superset(&lr_inno));
                assert!(lr.is_sorted_by_key(|c| c.inno));
                for ref lr_conn in lr {
                    match (l_map.get(&lr_conn.inno), r_map.get(&lr_conn.inno)) {
                        (None, None) => panic!("{} is in neither l nor r", lr_conn.inno),
                        (None, Some(conn)) => panic!("{} is in only r", conn.inno),
                        (Some(conn), None) => {
                            assert_from_connection!(lr_conn, *conn)
                        }
                        (Some(l_conn), Some(r_conn)) => {
                            assert_from_connection!(lr_conn, (*l_conn, *r_conn))
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_crossover_ne() {
        let l = [
            connection!(inno = 0, from = 1_1),
            connection!(inno = 1, from = 1_2),
            connection!(inno = 2, from = 1_3),
            connection!(inno = 9, from = 1_4),
        ];
        let r = [
            connection!(inno = 0, from = 2_1),
            connection!(inno = 2, from = 2_2),
            connection!(inno = 3, from = 2_3),
            connection!(inno = 4, from = 2_4),
            connection!(inno = 7, from = 2_5),
        ];

        assert_crossover_ne(&l, &r);
    }

    #[test]
    fn test_crossover_ne_empty() {
        let l = [connection!(inno = 0, from = 1_1)];

        assert_crossover_ne(&l, &[]);
        assert_crossover_ne(&[], &[]);
    }

    #[test]
    fn test_crossover_ne_no_overlap() {
        let l = [
            connection!(inno = 1, from = 1_1),
            connection!(inno = 3, from = 1_2),
            connection!(inno = 5, from = 1_3),
        ];
        let r = [
            connection!(inno = 0, from = 2_1),
            connection!(inno = 2, from = 2_2),
            connection!(inno = 4, from = 2_3),
        ];

        assert_crossover_ne(&l, &r);
    }

    #[test]
    fn test_crossover_ne_full_overlap() {
        let l = [
            connection!(inno = 1, from = 1_1),
            connection!(inno = 2, from = 1_2),
            connection!(inno = 3, from = 1_3),
        ];
        let r = [
            connection!(inno = 1, from = 2_1),
            connection!(inno = 2, from = 2_2),
            connection!(inno = 3, from = 2_3),
        ];

        assert_crossover_ne(&l, &r);
    }

    #[test]
    fn test_crossover_ne_overflow() {
        let l = [connection!(inno = 10, from = 1_1)];
        let r = [
            connection!(inno = 1, from = 2_1),
            connection!(inno = 2, from = 2_2),
        ];

        assert_crossover_ne(&l, &r);
    }

    #[test]
    fn test_crossover_ne_no_lt() {
        let l = [connection!(inno = 0, from = 1_1)];
        let r = [connection!(inno = 10, from = 2_1)];

        assert_crossover_ne(&l, &r);
    }

    #[test]
    fn test_crossover_lt() {
        let l = [
            connection!(inno = 0, from = 1_1),
            connection!(inno = 1, from = 1_2),
            connection!(inno = 2, from = 1_3),
        ];
        let r = [
            connection!(inno = 0, from = 2_1),
            connection!(inno = 2, from = 2_2),
            connection!(inno = 3, from = 2_3),
            connection!(inno = 4, from = 2_4),
        ];

        assert_crossover_ne(&l, &r);
        for (le, ge) in crossover(&l, &r, Ordering::Less, &mut rng())
            .iter()
            .zip(crossover_ne(&r, &l, &mut rng()))
        {
            assert_eq!(le.inno, ge.inno);
        }
    }
}
