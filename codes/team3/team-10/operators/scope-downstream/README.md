# Downstream microscope

This operator acts as a stand-in for the real microscope (the first operator). We did not have enough time to "wrap around" the loop. Further, we tried removing the microscope process (via multiprocessing) and directly connect to the other microscope instance created in the first operator. However, we got the following error:

```sh
Error in kernel: the calling thread is not the owner of this proxy, create a new proxy in this thread or transfer ownership.
```

In the future, we would have a mechanism to correctly notify the first operator (the "microscope").

We also started working on another functionality to the first operator, found in [operators/scope/daq-agent.py](../operators/scope/daq-agent.py), which would go to a location using the `DTMicroscope` API and make a scan. For this demo, we have just re-initialized another server with a copy of the original data, and shown a 20 pixel area around the "interesting" sample coordinate (based on the upstream analysis).
