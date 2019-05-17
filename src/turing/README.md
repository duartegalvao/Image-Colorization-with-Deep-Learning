# Turing Test

This is a simple test to manually assess the quality of the results produced by the neural network.

## Running

To run this, just run a simple HTTP server on this directory.

For example, if you have `php` installed, run:
```bash
php -S localhost:8001
```
And the test is available on `http://localhost:8001/`.

The images should be located in the `src/turing/images folder`.

You can also run a python SimpleHTTPServer, for the same effect:

```bash
python -m SimpleHTTPServer 8001
```

## Statistics script

Also included is a python script to parse the test results and calculate relevant statistics.

Simply run it as follows:

```bash
python3 stats.py <filename>
```

Where `<filename>` is the path to the generated JSON file.
