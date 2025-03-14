#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu, specie::population_init, Ctrnn, EvolutionTarget, Genome, Network, Scenario,
    Specie,
};
use nes_rust::{
    button::Button, default_audio::DefaultAudio, default_display::DefaultDisplay,
    default_input::DefaultInput, rom::Rom, Nes,
};

#[rustfmt::skip]
mod v {
pub const ID: usize           = 0x42;
pub const ID_NEXT: usize      = 0x19;
pub const X: usize            = 0x40;
pub const Y: usize            = 0x41;
pub const SPEED: usize        = 0x44;
pub const FALLTIME: usize     = 0x45;
pub const GAME_MODE: usize    = 0xc0;
pub const GAME_OVER: usize    = 0x58;
pub const SEED_L: usize       = 0x17;
pub const SEED_R: usize       = 0x18;
pub const SCORE_1: usize      = 0x53;
pub const SCORE_2: usize      = 0x54;
pub const SCORE_3: usize      = 0x55;
pub const PIECE_COUNT: usize  = 0x1a;

pub const BOARD_OFFSET: usize = 0x400;
pub const BOARD_SIZE: usize   = 200;

pub const INPUT_SIZE: usize   = 200;

pub const PIECE_SHAPE: [[(u8, u8); 4]; 19] = [
    [(3, 2), (4, 1), (4, 2), (4, 3)], // T_UP
    [(1, 2), (2, 2), (2, 3), (3, 2)], // T_RIGHT
    [(2, 1), (2, 2), (2, 3), (3, 2)], // T_DOWN
    [(1, 2), (2, 1), (2, 2), (3, 2)], // T_LEFT
    [(1, 2), (2, 2), (3, 1), (3, 2)], // J_UP
    [(2, 1), (3, 1), (3, 2), (3, 3)], // J_RIGHT
    [(1, 2), (1, 3), (2, 2), (3, 2)], // J_DOWN
    [(2, 1), (2, 2), (2, 3), (3, 3)], // J_LEFT
    [(2, 1), (2, 2), (3, 2), (3, 3)], // Z_HORIZONTAL
    [(1, 3), (2, 2), (2, 3), (3, 2)], // Z_VERTICAL
    [(2, 1), (2, 2), (3, 1), (3, 2)], // O
    [(2, 2), (2, 3), (3, 1), (3, 2)], // S_HORIZONTAL
    [(1, 2), (2, 2), (2, 3), (3, 3)], // S_VERTICAL
    [(1, 2), (2, 2), (3, 2), (3, 3)], // L_UP
    [(2, 1), (2, 2), (2, 3), (3, 1)], // L_RIGHT
    [(1, 1), (1, 2), (2, 2), (3, 2)], // L_DOWN
    [(2, 3), (3, 1), (3, 2), (3, 3)], // L_LEFT
    [(0, 2), (1, 2), (2, 2), (3, 2)], // I_VERTICAL
    [(3, 0), (3, 1), (3, 2), (3, 3)], // I_HORIZONTAL
];
}

use v::*;
fn sense_board(ram: &[u8], sense: &mut [f64; INPUT_SIZE]) {
    *sense = [0.; INPUT_SIZE];
    for (idx, _) in ram[BOARD_OFFSET..BOARD_OFFSET + BOARD_SIZE]
        .iter()
        .enumerate()
        .filter(|(_, b)| **b != 0xef)
    {
        sense[idx] = 1.;
    }

    if (0..19).contains(&ram[ID]) {
        for index in PIECE_SHAPE[ram[ID] as usize]
            .iter()
            .filter_map(|(row, col)| {
                let row = row + ram[Y];
                let col = col + ram[X];
                (row >= 2 && col >= 2).then(|| (((row - 2) as usize * 10) + (col - 2) as usize))
            })
            .filter(|index| *index < 200)
        {
            sense[index] = -1.;
        }
    } else {
        // TODO what piece am I missing??????
    }
}

fn score(ram: &[u8]) -> f64 {
    // real score | piece count
    // (((ram[SCORE_1] as usize) << 8)
    //     | ((ram[SCORE_2] as usize) << 16)
    //     | ((ram[SCORE_3] as usize) << 24)
    //     | (ram[PIECE_COUNT] as usize)) as f64

    ram[PIECE_COUNT] as f64
}

#[cfg(feature = "watch_game")]
fn draw_sense(sense: &[f64; INPUT_SIZE]) {
    for chunk in sense.chunks(10) {
        println!(
            "{}",
            chunk
                .iter()
                .map(|data| match data {
                    -1. => '-',
                    0. => '_',
                    1. => '+',
                    _ => '?',
                })
                .collect::<String>()
        )
    }
}

#[cfg(feature = "watch_game")]
fn draw_act(act: &[bool; 8]) {
    for b in act {
        print!("{} ", if *b { 'x' } else { '_' })
    }
    println!("\na b - + ^ . < > \n")
}

fn enter_game(nes: &mut Nes) {
    while nes.get_cpu().get_ram().data[0xc3] == 0 {
        nes.step_frame();
    }
    nes.get_mut_cpu().get_mut_ram().data[0xc3] = 0;
    while nes.get_cpu().get_ram().data[GAME_MODE] == 0 {
        nes.step_frame();
    }
    while nes.get_cpu().get_ram().data[GAME_MODE] != 4 {
        nes.press_button(Button::Start);
        nes.step_frame();
        nes.release_button(Button::Start);
        nes.step_frame();
    }

    nes.get_mut_cpu().get_mut_ram().data[SEED_L] = 0;
    nes.get_mut_cpu().get_mut_ram().data[SEED_R] = 0;
}

struct NesTetris;

impl Scenario for NesTetris {
    fn io(&self) -> (usize, usize) {
        (200, 8)
    }

    fn eval<F: Fn(f64) -> f64>(&self, genome: &Genome, σ: F) -> f64 {
        let mut nes = Nes::new(
            Box::new(DefaultInput::new()),
            Box::new(DefaultDisplay::new()),
            Box::new(DefaultAudio::new()),
        );
        nes.set_rom(Rom::new(include_bytes!("data/tetris.nes").to_vec()));
        nes.bootup();
        enter_game(&mut nes);

        let mut network = Ctrnn::from_genome(genome);
        let mut sense = [0.; 200];
        while nes.get_cpu().get_ram().data[GAME_OVER] == 0 {
            sense_board(&nes.get_cpu().get_ram().data, &mut sense);
            network.step(1, &sense, &σ);

            for (idx, x) in network.output().iter().enumerate() {
                if idx == 2 || idx == 3 {
                    continue;
                }
                nes.get_mut_cpu().joypad1.buttons[idx] = *x >= 0.5;
            }
            nes.step_frame();

            #[cfg(feature = "watch_game")]
            {
                print!("{}[2J", 27 as char);
                draw_sense(&sense);
                draw_act(&nes.get_cpu().joypad1.buttons);
                println!("{}", score(&nes.get_cpu().get_ram().data),);
            }

            nes.get_mut_cpu().joypad1.buttons = [false; 8];
        }

        score(&nes.get_cpu().get_ram().data)
    }
}

const POPULATION: usize = 100;

fn main() {
    let res = NesTetris {}.evolve(
        EvolutionTarget::Generation(10),
        |(i, o)| population_init(i, o, POPULATION),
        POPULATION,
        relu,
    );

    println!(
        "top score: {:?}",
        res.0
            .into_iter()
            .flat_map(|Specie { members, .. }| members)
            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
            .unwrap()
            .1
    );
}
