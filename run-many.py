#!/usr/bin/python3

import os
import json
import time
import socket
import argparse
import importlib
import statistics
import subprocess
import collections
import multiprocessing
import typing

scan = importlib.import_module("run-scan")

RunningProgram = collections.namedtuple("RunningProgram", ["program", "index", "handle"])

class Program:
    def __init__(self, description: str, opts: argparse.Namespace):
        # Separate the different options of the program
        s = description.split(":")
        s2 = s[0].split(" ")
        self._program = s2[0]
        self._programArgs = s2[1:]
        self._options = ["events", "eventsPerStream", "threads",
                         "streams","numa","cores","cudaDevices"]
        valid = set(self._options)
        for o in s[1:]:
            # For every option check that it is valid, and if it is remove it from the set
            # of valid options, otherwise raise an exception
            (name, value) = o.split("=")
            if name not in valid:
                raise Exception("Unsupported option '{}'".format(name))
            setattr(self, "_"+name, value)
            valid.remove(name)
        # The remaining valid options have not been specified, so set them to None
        for o in valid:
            setattr(self, "_"+o, None)

        # Set the number of events to run
        if opts.runForMinutes > 0:
            if self._events is not None or self._eventsPerStream is not None:
                raise Exception("""--runForMinutes argument conflicts with
                                'events'/'eventsPerStream'""")
        elif self._events is None:
            self._events = 1000
        # If the user didn't set the number of threads to use, use 1
        if self._threads is None:
            self._threads = 1
        # If the user didn't set the number of streams, use one for each thread
        if self._streams is None:
            self._streams = self._threads
        if self._eventsPerStream is not None:
            self._events = int(self._eventsPerStream)*int(self._streams)
            self._eventsPerStream = None
        self._cudaDevices = self._cudaDevices.split(",") if self._cudaDevices is not None else []

    def program(self) -> str:
        """Return the path to the test program to run"""
        return self._program

    def programShort(self) -> str:
        """Return the basename of the path to the program"""
        return os.path.basename(self._program)

    def events(self) -> int:
        """Return the number of events"""
        return self._events

    def cudaDevices(self) -> list:
        """Return the list of cuda devices"""
        return self._cudaDevices

    def makeCommandMessage(self, opts: argparse.Namespace) -> tuple:
        """
        Create a message for the program containing the positional and optional arguments
        and their values
        """
        command = [self._program] + self._programArgs
        if opts.runForMinutes > 0:
            command.extend(["--runForMinutes", str(opts.runForMinutes)])
        else:
            command.extend(["--maxEvents", str(self._events)])
        command.extend(["--numberOfThreads", str(self._threads),
                        "--numberOfStreams", str(self._streams)])
        msg = "Program {} threads {} streams {}".format(self.programShort(),
                                                        self._threads,
                                                        self._streams)
        if self._numa is not None:
            command = ["numactl", "--cpunodebind={}".format(self._numa),
                       "--membind={}".format(self._numa)] + command
            msg += " NUMA node {}".format(self._numa)
        if self._cores is not None:
            command = ["taskset", "-c", self._cores] + command
            msg += " cores {}".format(self._cores)
        if len(self._cudaDevices) > 0:
            msg += " CUDA devices {}".format(",".join(self._cudaDevices))
        return (command, msg)

    def makeEnv(self, logfile: typing.IO) -> dict:
        if len(self._cudaDevices) == 0:
            return os.environ  # dict
        visibleDevices = ",".join(self._cudaDevices)
        logfile.write("export CUDA_DEVICE_ORDER=PCI_BUS_ID\n")
        logfile.write("export CUDA_VISIBLE_DEVICES="+visibleDevices+"\n")
        logfile.flush()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=visibleDevices)
        return env

    def addMetadata(self, d: dict) -> None:
        d["program"] = self.programShort()
        d["threads"] = int(self._threads)
        d["streams"] = int(self._streams)
        if self._numa is not None:
            d["numa"] = int(self._numa)
        if self._cores is not None:
            d["cores"] = self._cores
        if self._cudaDevices is not None:
            d["cudaDevices"] = self._cudaDevices

    def __str__(self) -> str:
        ret = self._program+": "
        for o in self._options:
            v = getattr(self, "_"+o)
            if v is not None:
                ret += o+"="+v+" "
        return ret

class Monitor:
    def __init__(self, opts: argparse.Namespace, programs: list, cudaDevices: list = []):
        self._intervalSeconds = opts.monitorSeconds
        self._monitorMemory = opts.monitorMemory
        self._monitorClock = opts.monitorClock
        self._monitorCuda = opts.monitorCuda
        self._allPrograms = programs

        self._timeStamp = []
        self._dataMemory = [[] for _ in programs]
        self._dataClock = {x: [] for x in range(0, multiprocessing.cpu_count())}
        self._dataCuda = {x: [] for x in cudaDevices}
        self._dataCudaProcs = [{x: [] for x in p.cudaDevices()} for p in programs]

    def setIntervalSeconds(self, interval: int) -> None:
        """Set the time interval for the data monitoring"""
        self._intervalSeconds = interval

    def intervalSeconds(self) -> int:
        """Return the time interval for the data monitoring"""
        return self._intervalSeconds

    def snapshot(self, programs: list = []) -> None:
        if self._intervalSeconds is None:
            return
        self._timeStamp.append(time.strftime("%y-%m-%d %H:%M:%S"))

        # Monitor host memory
        if self._monitorMemory:
            update = [dict(rss=0)]*len(self._dataMemory)
            for rp in programs:
                update[rp.index]["rss"] = scan.processRss(rp.handle.pid)
            for i, u in enumerate(update):
                self._dataMemory[i].append(u)
        # Monitor the CPU core clocks
        if self._monitorClock:
            clocks = scan.processClock()
            for key, lst in self._dataClock.items():
                lst.append(dict(clock=clocks.get(key, -1.0)))

        # Monitor utilization, power consumption and memory usage of CUDA devices
        if self._monitorCuda:
            for dev in self._dataCuda.keys():
                self._dataCuda[dev].append(scan.cudaDeviceStatus(dev)._asdict())
            update = [{x: dict(proc_mem_use=0) for x in p.cudaDevices()} for p in self._allPrograms]
            for rp in programs:
                for dev in rp.program.cudaDevices():
                    update[rp.index][dev]["proc_mem_use"] = \
                            scan.cudaDeviceProcessMemory(dev, rp.handle.pid)
            for i, u in enumerate(update):
                for dev, val in u.items():
                    self._dataCudaProcs[i][dev].append(val)

    def toArrays(self) -> dict:
        data = {}
        if self._intervalSeconds is not None:
            data["time"] = self._timeStamp
            if self._monitorMemory or self._monitorClock:
                data["host"] = {}
                if self._monitorMemory:
                    data["host"]["processes"] = self._dataMemory
                if self._monitorClock:
                    data["host"]["cpu"] = self._dataClock
            if self._monitorCuda:
                data["cuda"] = dict(
                    device = self._dataCuda,
                    processes = self._dataCudaProcs,
                )
        return data

def runMany(programs: list,
            opts: argparse.Namespace,
            logfilenamebase: str,
            monitor: Monitor) -> list:
    # Create a logfile for each of the programs to be run
    logfiles = []
    for i in range(0, len(programs)):
        logfiles.append(open(logfilenamebase.format(i), "w"))

    running_programs = []
    for i, (prog, logfile) in enumerate(zip(programs, logfiles)):
        (command, msg) = prog.makeCommandMessage(opts)
        msg = str(i) + " "+ msg
        # Append the number of minutes or number of events to the command message
        if opts.runForMinutes > 0:
            msg += " minutes {}".format(opts.runForMinutes)
        else:
            msg += " events {}".format(prog.events())
        # Print the current time and day, followed by the number of the program and
        # the options for it's execution
        scan.printMessage(msg)
        # In the logfile write the commands separated by a white space
        logfile.write(" ".join(command))
        logfile.write("\n----\n")
        logfile.flush()
        # If the user chose the dryRun option, print the commands
        if opts.dryRun:
            print(" ".join(command))
            continue

        env = prog.makeEnv(logfile)
        p = subprocess.Popen(command,
                             stdout=logfile,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True,
                             env=env)
        running_programs.append(RunningProgram(prog, i, p))
    monitor.snapshot(running_programs)
    finished_programs = []

    def terminate_programs() -> None:
        for p in running_programs:
            try:
                p.handle.terminate()
            except OSError:
                pass
            p.handle.wait()

    # While there are still some programs running, wait for them to finish for the chosen
    # time length, and if they finish in that time move them to the list of finished programs
    while len(running_programs) > 0:
        try:
            running_programs[0].handle.wait(timeout=monitor.intervalSeconds())
            rp = running_programs[0]
            del running_programs[0]
            finished_programs.append(rp)
            scan.printMessage(f"Program {rp.index} finished")
            if rp.handle.returncode != 0:
                print(f" got return code {rp.handle.returncode}, aborting test")
                msg += "Program {} {} got return code {}, see output in log file {}, terminating the remaining programs.\n".format(rp.index,
                                                                                                                                   rp.program.program(),
                                                                                                                                   rp.handle.returncode,
                                                                                                                                   logfilenamebase.format(rp.index))
                terminate_programs()
        except subprocess.TimeoutExpired:
            monitor.snapshot(running_programs)
        except KeyboardInterrupt:
            terminate_programs()
    monitor.snapshot(running_programs)
    msg = ""
    for i, p in enumerate(running_programs):
        if p.returncode != 0:
            msg += "Program {} {} got return code %d,see output in log file %s\n".format(i,
                                                                                         programs[i].program(),
                                                                                         logfilenamebase.format(i))
    if len(msg) > 0:
        raise Exception(msg)

    for l in logfiles:
        l.close()

    ret = []
    for i in range(len(programs)):
        fname = logfilenamebase.format(i)
        with open(fname) as logfile:
            ret.append(scan.throughput(logfile, fname))
    return ret


def main(opts: argparse.Namespace) -> None:
    # Split the descriptions of the programs to be run
    programs = []
    cudaDevicesInPrograms = set()
    for x in opts.programs.split(";"):
        # If the users doesn't put the number of copies inside square brakets at the beginning
        # of the program declaration, it is set to 1
        num = 1
        if x[0] == "[":
            s = x[1:].split("]")
            num = int(s[0])
            # x contains the positional arguments for the program
            x = s[1]
        for i in range(0, num):
            # Initialize an instance of 'Program' with the program descriptions and the
            # namespace of optional arguments. Then append it to the list of programs
            p = Program(x, opts)
            programs.append(p)
            # Add cuda devices to the set of used cuda devices
            cudaDevicesInPrograms.update(p.cudaDevices())
    cudaDevicesInPrograms = list(cudaDevicesInPrograms)

    cudaDevicesAvailable = scan.listCudaDevices()
    print("Found {} devices".format(len(cudaDevicesAvailable)))
    for i, d in cudaDevicesAvailable.items():
        print(" {} {} driver {}".format(i, d.name, d.driver_version))

    # If any of the required cuda devices is not in the list of available devices, raise an
    # exception
    for d in cudaDevicesInPrograms:
        if d not in cudaDevicesAvailable:
            raise Exception("Some program asked device {} but there is no device with that id".format(d))

    # Create and fill the dict that will be written in the json file
    data = dict(
        results=[]
    )
    outputJson = opts.output+".json"
    if os.path.exists(outputJson):
        if opts.append:
            with open(outputJson) as inp:
                data = json.load(inp)
        elif not opts.overwrite:
            return

    hostname = socket.gethostname()

    # Try for the maximum number of tries chosen by the user (1 by default)
    tryAgain = opts.tryAgain
    while tryAgain > 0:
        try:
            monitor = Monitor(opts, programs, cudaDevices=cudaDevicesInPrograms)
            measurements = runMany(programs, opts, opts.output+"_log_{}.txt", monitor=monitor)
            break
        except Exception as e:
            tryAgain -= 1
            if tryAgain == 0:
                raise
            print("Got exception (see below), trying again ({} times left)".format(tryAgain))
            print("--------------------")
            print(str(e))
            print("--------------------")

    d = dict(
        hostname=hostname,
        throughput=sum(m.throughput for m in measurements),
        cpueff=statistics.mean(m.cpueff for m in measurements) if len(measurements) > 0 else 0,
        programs=[]
    )
    scan.printMessage("Total throughput {} events/s".format(d["throughput"]))
    for i, (p, m) in enumerate(zip(programs, measurements)):
        dp = dict(
            events=m.events,
            time=m.time,
            throughput=m.throughput,
            cpueff=m.cpueff,
        )
        p.addMetadata(dp)
        d["programs"].append(dp)
    if monitor.intervalSeconds() is not None:
        d["monitor"]=monitor.toArrays()
    if len(cudaDevicesInPrograms) > 0:
        d["cudaDevices"] = {
            x: dict(name=cudaDevicesAvailable[x].name,
                    driver_version=cudaDevicesAvailable[x].driver_version) for x in cudaDevicesInPrograms
        }
    data["results"].append(d)

    with open(outputJson, "w") as out:
        json.dump(data, out, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Run given test programs.

Note that this program does not honor CUDA_VISIBLE_DEVICES, use cudaDevices instead.

Measuring combined throughput of multiple programs
[M]<program>:threads=N:streams=N:numa=N:cores=<list>:cudaDevices=<list>;<program>:...

  <program>       (Path to) the program to run
  events          Number of events to process (default 1000 if --runForMinutes is not specified)
  eventsPerStream Number of events per stream to process
  threads         Number host threads (default: 1)
  streams         Number of streams (concurrent events) (default: same as threads)
  numa            NUMA node, uses 'numactl' (default: not set)
  cores           List of CPU cores to pin, uses 'taskset' (default: not set)
  cudaDevices     List of CUDA devices to use (default: not set)
  M copies
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("programs", type=str,
                        help="Declaration of many programs to run (for syntax see above).")

    # Add the optional arguments for the output JSON file and for the monitoring
    # to the parser
    scan.addCommonArguments(parser)

    # Add the runForMinuts optional argument
    parser.add_argument("--runForMinutes", type=int, default=-1,
                        help="Process the set of events until this many minutes has elapsed."
                        + "Conflicts with 'events'. (default -1 for disabled)""")

    # Parse the positional and optional arguments and call the main function
    opts = scan.parseCommonArguments(parser)
    main(opts)
