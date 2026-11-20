#!/usr/bin/env bash
# Standalone JVM test runner (no Gradle required): compiles core + simulator with
# kotlinc and runs the JUnit console launcher.
#
# Prereqs: JDK 17+, kotlinc, and junit-platform-console-standalone.jar.
# Override locations via JAVA_HOME / KOTLIN_HOME / JUNIT_JAR env vars.
set -euo pipefail
cd "$(dirname "$0")/.."

KOTLINC="${KOTLIN_HOME:+$KOTLIN_HOME/bin/}kotlinc"
JUNIT_JAR="${JUNIT_JAR:-junit-platform-console-standalone.jar}"
STDLIB="${KOTLIN_HOME:+$KOTLIN_HOME/lib/kotlin-stdlib.jar}"

mkdir -p build/core build/core-test build/sim build/sim-test
"$KOTLINC" core/src/main/kotlin -d build/core
"$KOTLINC" -cp "build/core:$JUNIT_JAR" core/src/test/kotlin -d build/core-test
"$KOTLINC" -cp build/core simulator/src/main/kotlin -d build/sim
"$KOTLINC" -cp "build/core:build/sim:$JUNIT_JAR" simulator/src/test/kotlin -d build/sim-test

java -Djava.awt.headless=true -jar "$JUNIT_JAR" execute \
    -cp "build/core:build/core-test:build/sim:build/sim-test:$STDLIB" \
    --scan-classpath --details=summary

java -Djava.awt.headless=true -cp "build/core:build/sim:$STDLIB" \
    com.example.rotatex.sim.DemoMainKt demo_out
