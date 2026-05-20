#!/bin/bash
set -e
echo "=== Building PromeTorch for all Russian platforms ==="
for platform in astra alt redos elbrus baikal; do
    echo ""
    echo "=== Building for $platform ==="
    docker build -t prometorch-$platform -f docker/Dockerfile.$platform ..
    echo "=== $platform: BUILD SUCCESS ==="
done
echo ""
echo "=== All platforms built successfully ==="
