# Contributing

Contributions to the driver-monitoring prototype are welcome.

## Development setup

    git clone https://github.com/arianharashchi-lang/Driver_-monitoring.git
    cd Driver_-monitoring
    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    uvicorn webapp.app:app --reload


Before opening a pull request, check Python and JavaScript syntax and run the driver tests:

    python -m compileall -q .
    node --test tests/js/*.test.cjs


Do not commit credentials, private camera recordings, or identifying screenshots without consent. Document behavior changes and describe detection results as uncertain indicators.