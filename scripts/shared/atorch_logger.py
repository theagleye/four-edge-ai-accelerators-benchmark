# logs power from the atorch j7-c usb meter over bluetooth le, used for the coral and vim3
# runs on: host pc
# output: atorch_power_log.csv

# atorch j7-c BLE logger: power + temperature
# protocol: type 03, 36-byte packet, 1/sec
# voltage bytes [4:7] big-endian, unit 10mV -> /100 = V
# current bytes [7:10] big-endian, unit 10mA -> *10/1000 = A
# temp bytes [21:23] big-endian, unit 1 degC

import asyncio
import csv
import signal
import statistics
import sys
import time
from bleak import BleakClient, BleakScanner

DEV_NAME = 'UC96_BLE'
NOTIFY_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb'
SCAN_TO = 15.0  # scan timeout secs
MAX_RETRY = 10  # reconnection attempts
RETRY_DELAY = 3.0  # secs between retries
OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else 'atorch_power_log.csv'

readings = []  # (timestamp, V, A, W, degC)
running = True


def parse_pkt(data):
    if len(data) != 36:
        return None
    if data[0] != 0xff or data[1] != 0x55:
        return None
    if data[3] != 0x03:
        return None
    v_raw = (data[4] << 16) | (data[5] << 8) | data[6]
    i_raw = (data[7] << 16) | (data[8] << 8) | data[9]
    t_raw = (data[21] << 8) | data[22]
    V = v_raw / 100.0
    A = (i_raw * 10) / 1000.0
    W = V * A
    T = float(t_raw)
    return V, A, W, T


def on_notify(sender, data):
    parsed = parse_pkt(data)
    if parsed is None:
        print(f'warn: bad packet: {data.hex()}')
        return
    V, A, W, T = parsed
    ts = time.time()
    readings.append((ts, V, A, W, T))
    elapsed = ts - readings[0][0] if len(readings) > 1 else 0.0
    print(f'[{elapsed:8.1f}s]  {V:.3f}V  {A*1000:6.1f}mA  {W:.4f}W  {T:.0f}C')


def save_summary():
    if not readings:
        print('no data collected')
        return
    watts = [r[3] for r in readings]
    volts = [r[1] for r in readings]
    amps = [r[2] for r in readings]
    temps = [r[4] for r in readings]
    n = len(readings)
    dur = readings[-1][0] - readings[0][0]
    print('\nsummary:')
    print(f'samples  : {n}')
    print(f'duration : {dur:.1f}s')
    print(f'voltage  : mean={statistics.mean(volts):.3f}V  min={min(volts):.3f}V  max={max(volts):.3f}V')
    print(f'current  : mean={statistics.mean(amps)*1000:.1f}mA  min={min(amps)*1000:.1f}mA  max={max(amps)*1000:.1f}mA')
    print(f'power    : mean={statistics.mean(watts):.4f}W  min={min(watts):.4f}W  max={max(watts):.4f}W  std={statistics.stdev(watts) if n > 1 else 0:.4f}W')
    print(f'temp     : mean={statistics.mean(temps):.1f}C  min={min(temps):.0f}C  max={max(temps):.0f}C  rise={max(temps)-temps[0]:.0f}C')
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_s', 'voltage_V', 'current_A', 'power_W', 'temp_C'])
        for ts, V, A, W, T in readings:
            w.writerow([f'{ts:.3f}', f'{V:.3f}', f'{A:.4f}', f'{W:.4f}', f'{T:.0f}'])
    print(f'saved {n} rows -> {OUT_CSV}')


async def run():
    global running
    attempt = 0
    while running and attempt < MAX_RETRY:
        attempt += 1
        if attempt > 1:
            print(f'[reconnect] attempt {attempt}/{MAX_RETRY}, waiting {RETRY_DELAY}s ...')
            await asyncio.sleep(RETRY_DELAY)
        print(f'scanning for {DEV_NAME} ...')
        try:
            dev = await BleakScanner.find_device_by_name(DEV_NAME, timeout=SCAN_TO)
        except Exception as e:
            print(f'warn: scan error: {e}')
            continue
        if dev is None:
            print(f'warn: {DEV_NAME} not found, retrying ...')
            continue
        print(f'found: {dev.name} [{dev.address}]')
        try:
            async with BleakClient(dev, timeout=15.0) as client:
                if not client.is_connected:
                    print('warn: connect failed, retrying ...')
                    continue
                attempt = 0
                if len(readings) == 0:
                    print(f'connected | output: {OUT_CSV} | ctrl+c to stop\n')
                    print(f"{'elapsed':>9}  {'V':>7}  {'mA':>8}  {'W':>9}  {'C':>5}")
                else:
                    print(f'[reconnect] ok - {len(readings)} samples kept')
                await client.start_notify(NOTIFY_UUID, on_notify)
                while running:
                    await asyncio.sleep(0.5)
                    if not client.is_connected:
                        print('warn: ble dropped')
                        break
                try:
                    await client.stop_notify(NOTIFY_UUID)
                except Exception:
                    pass
        except Exception as e:
            print(f'warn: ble error: {e}')
            if not running:
                break
            continue
    if attempt >= MAX_RETRY and running:
        print(f'[error] gave up after {MAX_RETRY} attempts')


def stop(sig, frame):
    global running
    print('\nstopping ...')
    running = False


signal.signal(signal.SIGINT, stop)

if __name__ == '__main__':
    try:
        asyncio.run(run())
    finally:
        save_summary()
