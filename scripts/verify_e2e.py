"""After training: validate reload, held-out prediction, pytest and local HTTP health."""
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from src.data_processing import load_dataset
    from src.predict import load_bundle, predict_patient
    bundle = load_bundle(ROOT / 'models/heart_models.joblib')
    data = load_dataset(ROOT / 'data/heart.csv')
    sample_index = bundle['metadata']['split_indices']['test'][0]
    result = predict_patient(data.loc[sample_index].to_dict(), bundle)
    for prediction in result['predictions'].values():
        assert np.isfinite(prediction['probability']) and 0 <= prediction['probability'] <= 1
    print('Saved models predict successfully on an actual held-out record.', flush=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True)
    # Local address only. Always terminate the server, including failed checks.
    with tempfile.TemporaryFile(mode='w+') as log:
        server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
                                   '--server.address', '127.0.0.1', '--server.port', '8507',
                                   '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError('Streamlit exited before becoming healthy')
                try:
                    with urlopen('http://127.0.0.1:8507/_stcore/health', timeout=1) as response:
                        assert response.read() == b'ok'
                        print('Streamlit local health endpoint: OK')
                        break
                except OSError:
                    time.sleep(.25)
            else:
                raise RuntimeError('Streamlit health endpoint did not respond within 15 seconds')
        except Exception:
            log.seek(0)
            print(log.read())
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    print('End-to-end verification complete.')


if __name__ == '__main__':
    main()
