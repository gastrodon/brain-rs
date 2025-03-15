FROM rust:alpine AS builder
RUN apk add --no-cache musl-dev

WORKDIR /opt/brain-rs
ADD src src
ADD examples examples
ADD benches benches
ADD Cargo.toml Cargo.lock .
RUN cat Cargo.toml


ARG TARGET
ARG BUILD_OPT="-r"
RUN cargo build ${BUILD_OPT} --example ${TARGET}

FROM alpine:latest
WORKDIR /opt/brain-rs

ARG TARGET
COPY --from=builder /opt/brain-rs/target/release/examples/${TARGET} /opt/brain-rs/run

RUN mkdir -p genome
ENTRYPOINT /opt/brain-rs/run
