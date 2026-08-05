FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    # libdmtx: REQUIRED by pylibdmtx, which generates the DataMatrix codes.
    # pylibdmtx publishes no Linux wheel — only a pure-Python one plus Windows
    # wheels that bundle the DLL. On Linux it resolves the library at runtime
    # with ctypes.util.find_library('dmtx') and raises
    #   ImportError: Unable to find dmtx shared library
    # if it is missing. That failure is invisible at startup: the import-time
    # layout probe in cdp_engine catches it and falls back to hardcoded
    # constants, so the container boots and /api/health returns ok while every
    # label-pdf and label-png request returns HTTP 500.
    libdmtx0b \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium

COPY . .

RUN mkdir -p patterns uploads data

EXPOSE 8000

CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "--workers", "2"]
