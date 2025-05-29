# Usage of interactEM

There are 5 operators here. Each of the directories below has the following format:

1. `operator.json` --  this is the `json` representation of an operator. This is ingested by our frontend to display the operator. Eventually, we will do away with this concept for `interactEM`, and this information will be represented by container labels.
2. `Containerfile` -- each of these containerfiles builds on the operator containerfile from the `interactEM` repository. It contains all of the necessary backend stuff to get started.
3. `run.py` -- this is the actual operator. We have not fully fleshed out our API in `interactEM`, but the basic premise is that the `@operator` decorator creates a function that will operate on incoming data (from a `zeromq` stream) and send it out to downstream operators. The `@dependencies` decorator will bring up servers, or whatever is needed for the operator to work.
4. `build.sh` contains the script to build the operator.

These operators were built on a Mac M1 and pulled into a local instance of `interactEM` to run. In principle, these operators could be deployed anywhere. For the purposes of this demo, it is only local. `interactEM` is __very__ alpha at the moment, but you are welcome to explore the repo for the source.
