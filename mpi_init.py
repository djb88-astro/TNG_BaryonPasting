import numpy as np
from mpi4py import MPI

"""
This class combines various MPI functions together, which makes it
easier to call them and provides pre-written examples for students
to see.
"""


class mpi:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.NProcs = self.comm.Get_size()
        self.Rank = self.comm.Get_rank()
        self.Name = MPI.Get_processor_name()

        self.get_number_of_nodes()
        return

    def get_number_of_nodes(self):
        """
        Using processor names to get the number of nodes

        NOTE: This works on Odyssey, but definately architecture dependent!!!!!
        """

        tmp = np.array([int(self.Name[6:11])])
        tmp = self.gather(tmp)
        tmp = np.unique(tmp)
        self.NNodes = len(tmp)
        del tmp
        return

    def gather(self, data):
        """
        Gathers an array on to all tasks

        Arguments:
          -data : Array to be gathered
        """

        # Get array shape
        shape = data.shape
        if len(shape) > 1:
            data = data.flatten("C")

        # Communicate expected array size and counts
        tmp = np.zeros(self.NProcs, dtype=np.int)
        tmp[self.Rank] = len(data)
        cnts = np.zeros(self.NProcs, dtype=np.int)
        self.comm.Allreduce([tmp, MPI.INT], [cnts, MPI.INT], op=MPI.SUM)
        del tmp

        # Calculate displacement for each task
        dspl = np.zeros(self.NProcs, dtype=np.int)
        dspl[1:] = np.cumsum(cnts[:-1])

        # MPI4PY v2+ requires conversion to float64 if not int
        tmp = data.dtype
        if tmp == "float32":
            data = data.astype(np.dtype("float64"))
        rslt = np.zeros(np.sum(cnts), dtype=data.dtype)

        # Gather array
        self.comm.Allgatherv(
            [data, cnts[self.Rank]], [rslt, cnts, dspl, MPI._typedict[data.dtype.char]]
        )

        rslt = rslt.astype(tmp)

        # Reshape if required
        if len(shape) > 1:
            rslt = rslt.reshape((-1, shape[1]))
        del data, shape, cnts, dspl, tmp
        return rslt

    def gatherv_single(self, data, root=0):
        """
        Gather all data to a chosen task routine

        Arguments:
          -data : Array to be gathered
          -root : Task on which to gather array [INT]
        """

        # Reshape to 1D if required
        shape = data.shape
        if len(shape) > 1:
            data = data.flatten("C")

        # Get counts on destination task
        tmp = np.zeros(self.NProcs, dtype=np.int)
        tmp[self.Rank] = len(data)
        cnts = np.zeros(self.NProcs, dtype=np.int)
        self.comm.Reduce([tmp, MPI.INT], [cnts, MPI.INT], op=MPI.SUM, root=root)
        del tmp

        # Create local receive array on destination, else None
        if self.Rank == root:
            rslt = np.empty(cnts.sum(), dtype=data.dtype)
        else:
            rslt = None

        # Now gather array on desination
        self.comm.Gatherv(sendbuf=data, recvbuf=(rslt, cnts), root=root)

        # Reshape if required
        if self.Rank == root:
            if len(shape) > 1:
                rslt = rslt.reshape((-1, shape[1]))
        return rslt

    def reduce(self, data):
        """
        Combine an array on all tasks

        Arguments:
          -data : Array to be reduced
        """

        # Check for scalar value
        if data.ndim == 0:
            data = np.atleast_1d(data)

        # Flatten multidimensional arrays
        shape = data.shape
        if len(shape) > 1:
            data = data.flatten("C")

        # MPI4PY v2+ requires conversion to float64 if not int
        tmp = data.dtype
        if tmp == "float32":
            data = data.astype(np.dtype("float64"))
        rslt = np.zeros(len(data), dtype=data.dtype)

        # Communicate
        self.comm.Reduce(
            [data, MPI._typedict[data.dtype.char]],
            [rslt, MPI._typedict[data.dtype.char]],
            op=MPI.SUM,
            root=0,
        )
        rslt = rslt.astype(tmp)

        # Reshape array if required
        if len(shape) > 1:
            rslt = rslt.reshape(shape)
        del data, shape, tmp
        return rslt
