# Turing Test

This is a simple test to manually assess the quality of the results produced by the neural network.

## Running

To run this, just run a simple HTTP server on the root directory of the project, and access the `/turing` URI.

For example, if you have `php` installed, run:
```bash
php -S localhost:8001
```
And the test is available on `http://localhost:8001/turing`.

You can also run a python SimpleHTTPServer, for the same effect:

```bash
python -m SimpleHTTPServer 8001
```
