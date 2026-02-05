#!/bin/bash
# Quick verification script for MNIST WebAssembly Demo
# Subtask 5-2: End-to-end test verification

echo "🧪 MNIST WebAssembly Demo - Quick Verification"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -d "demo" ] || [ ! -d "wasm" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

echo "1. Checking file structure..."
REQUIRED_FILES=(
    "demo/index.html"
    "demo/app.js"
    "demo/wasm_wrapper.js"
    "demo/model_loader.js"
    "demo/style.css"
    "demo/mnist_model.bin"
    "demo/pkg/mnist_wasm.js"
    "demo/pkg/mnist_wasm_bg.wasm"
    "demo/e2e_test.html"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "❌ Some files are missing. Please run the setup first."
    exit 1
fi

echo ""
echo "2. Checking if server is running on port 8080..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/index.html | grep -q "200"; then
    echo "   ✅ Server is running and accessible"
    SERVER_RUNNING=true
else
    echo "   ⚠️  Server not detected on port 8080"
    echo "   💡 Start with: cd demo && python3 -m http.server 8080"
    SERVER_RUNNING=false
fi

if [ "$SERVER_RUNNING" = true ]; then
    echo ""
    echo "3. Checking resource accessibility..."

    RESOURCES=(
        "/index.html"
        "/app.js"
        "/wasm_wrapper.js"
        "/model_loader.js"
        "/style.css"
        "/mnist_model.bin"
        "/pkg/mnist_wasm.js"
        "/pkg/mnist_wasm_bg.wasm"
        "/e2e_test.html"
    )

    for resource in "${RESOURCES[@]}"; do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080${resource}")
        if [ "$STATUS" = "200" ]; then
            echo "   ✅ $resource (HTTP $STATUS)"
        else
            echo "   ❌ $resource (HTTP $STATUS)"
        fi
    done
fi

echo ""
echo "=============================================="
echo "✅ Automated verification complete!"
echo ""
echo "📋 Next steps for manual testing:"
echo ""
echo "1. If server is not running, start it:"
echo "   cd demo"
echo "   python3 -m http.server 8080"
echo ""
echo "2. Open in browser:"
echo "   Main demo:  http://localhost:8080/index.html"
echo "   Test suite: http://localhost:8080/e2e_test.html"
echo ""
echo "3. Follow test steps in demo/E2E_TEST_REPORT.md"
echo ""
echo "4. Verify:"
echo "   ✓ Draw a digit on canvas"
echo "   ✓ Prediction bars update"
echo "   ✓ Top prediction highlighted"
echo "   ✓ Clear button works"
echo "   ✓ Test multiple digits"
echo ""

if [ "$SERVER_RUNNING" = true ]; then
    echo "🌐 Quick open commands:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   open http://localhost:8080/index.html"
        echo "   open http://localhost:8080/e2e_test.html"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "   xdg-open http://localhost:8080/index.html"
        echo "   xdg-open http://localhost:8080/e2e_test.html"
    fi
    echo ""
fi

echo "📖 Full test report: demo/E2E_TEST_REPORT.md"
echo "📖 Verification docs: demo/VERIFICATION_COMPLETE.md"
echo ""
