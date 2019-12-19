#include <iostream>
#include <mpi.h>

#include <boost/filesystem.hpp>
#include <boost/numeric/odeint.hpp>
#include <proto/protobuf_config.pb.h>
#include "global/GlobalDefinitions.h"

#include "util/ArgParser.h"
#include "logging/TimeLogger.h"
#include "factory/Factory.h"
#include "util/JsonToProtobufConfigConverter.h"
#include "logging/AsyncBuffer.h"

using namespace global_definitions;
using namespace boost::numeric::odeint;
using namespace std;

void observer(storage_type x, double currentTime, int numNeurons, int bufferSize, AsyncBuffer* buffer)
{
  storage_type toWrite(bufferSize);
  toWrite[0] = currentTime;
  for (int i = 0; i < numNeurons; i++) {
    toWrite[i+1] = x[i];
  }
  buffer->writeData(&(toWrite[0]));
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int mpiRank, mpiSize;
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  TimeLogger &tLogger = TimeLogger::getInstance();
  if (mpiRank == 0)
    tLogger.recordProgramStartTime();

  po::variables_map vm;
  if (!argparser::parse(argc, argv, vm)) {
    MPI_Finalize();
    return 0;
  }

  const auto &inputFile = vm["input-file"].as<string>();
  const auto &outputFile = vm["output-file"].as<string>();
  // FIXME: Hack for now until I figure out what C++ would prefer
  // I want this so I don't have multiple processes writing to the same file.
  // Ideally, we'll only have one output file, written to by rank 0.
  stringstream stream;
  stream << outputFile << mpiRank;
  const auto &outputFilename = stream.str();

  if (mpiRank == 0)
    tLogger.recordLoadConfigStartTime();

  JsonToProtobufConfigConverter converter;
  Config config = converter.readConfig(const_cast<string &>(inputFile));

  // const auto totalNeurons = config.neurons_size();
  // const auto totalSynapses = config.neurons_size();

  // // FIXME: Hard-coded like an idiot
  // int neuronCounts[2] = {1, 1};
  // int neuronAssignments[2][1] = {{0}, {1}};
  // int synapseCounts[2] = {1, 1};
  // int synapseAssignments[2][1] = {{1}, {0}}; //synapse 0 goes from N2 to N1

  //FIXME: Don't hard-code this either, you idiot
  if (mpiRank == 0)
  {
    config.mutable_neurons()->DeleteSubrange(1,1); // rank 0 doesn't need neuron 1
    config.mutable_synapses()->DeleteSubrange(0,1); // rank 0 doesn't need synapse 0
  }
  else
  {
    config.mutable_neurons()->DeleteSubrange(0,1);
    config.mutable_synapses()->DeleteSubrange(1,1);
  }

  // Now that we've pruned the protobuf config, we'll initialize only the items belonging
  // to our rank into the ProgramConfig, which will be referenced by the equation.
  // Notably, we still have vestiges of the deleted items in neurons' "incoming" arrays
  // and synapses' "source" values. It is paramount to handle these properly in the equation.

  config::ProgramConfig &c = config::ProgramConfig::getInstance();
  try {
    c.loadProtobufConfig(config);
  } catch (exception &e) {
    cerr << e.what() << endl;
    MPI_Finalize();
    return 1;
  }
  if (mpiRank == 0)
    tLogger.recordLoadConfigEndTime();

  const auto numNeuron = config.neurons_size();
  const int bufferSize = numNeuron + 1;
  const int precision = vm["output-precision"].as<int>();
  const int verbosity = vm["verbose-level"].as<int>();
  auto buffer = new AsyncBuffer(bufferSize, const_cast<string &>(outputFilename), precision, verbosity);

  sequential::ode_system_function *equation = factory::equation::getEquation(vm);

  storage_type x = c.getInitialStateValues();
  auto myStepper = make_controlled(c.absoluteError, c.relativeError, runge_kutta_dopri5<storage_type>());
  auto timeData = (double*) malloc(2*sizeof(double));
  double observerStep = 0.00025;
  timeData[0] = c.startTime;
  timeData[1] = observerStep;

  if (mpiRank == 0)
    tLogger.recordCalculationStartTime();

  int steps = 0;
  while ((timeData[0]+observerStep) <= c.endTime)
  {
    observer(x, timeData[0], numNeuron, bufferSize, buffer);

    // We only want to integrate one observerStep at a time.
    double targetTime = timeData[0] + observerStep;

    while (timeData[0] < targetTime)
    {
      if (targetTime < (timeData[0] + timeData[1])) // guarantee that we hit target_time exactly
        timeData[1] = targetTime - timeData[0];

      controlled_step_result result; // either success or fail
      do
      {
        ode::hodgkinhuxley::HodgkinHuxleyEquation::setSpeculative(false);
        result = myStepper.try_step(equation, x, timeData[0], timeData[1]);
      } while (result == fail);
    }

    // We do this rather than keeping the timeData[0] values that is set by try_step
    // upon a successful completion of a step because that may compound floating-point error.
    // Boost does this same thing with their integrate_const function.
    steps++;
    timeData[0] = c.startTime + static_cast<double>(steps)*observerStep;
    // I belive that it's actually more accurate to leave the try_step output as-is, but this
    // produces the same output as integrate_const.
    //MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Make an observation at t=c.endTime
  observer(x, timeData[0], numNeuron, bufferSize, buffer);

  if (mpiRank == 0)
    tLogger.recordCalculationEndTime();

  free(timeData);
  delete buffer;

  if (mpiRank == 0)
  {
    tLogger.recordProgramEndTime();
    tLogger.printSummary();
  }

  MPI_Finalize();

  return 0;
}
