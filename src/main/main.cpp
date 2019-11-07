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
    config.mutable_synapses()->DeleteSubrange(0,1); // rank 0 doesn't need synapse 9
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

  if (mpiRank == 0)
    tLogger.recordCalculationStartTime();
  integrate_const(
    // make_controlled(
    //   c.absoluteError,
    //   c.relativeError,
    //   runge_kutta_dopri5<storage_type>()
    // ),
    runge_kutta4<storage_type>(),
    equation,
    c.getInitialStateValues(),
    c.startTime,
    c.endTime,
    0.00025,
    [&](const storage_type &x, const double t) {
      storage_type toWrite(bufferSize);
      toWrite[0] = t;
      for (int i = 0; i < numNeuron; i++) {
        toWrite[i+1] = x[i];
      }
      buffer->writeData(&(toWrite[0]));
    }
  );
  if (mpiRank == 0)
    tLogger.recordCalculationEndTime();

  delete buffer;

  if (mpiRank == 0)
  {
    tLogger.recordProgramEndTime();
    tLogger.printSummary();
  }

  MPI_Finalize();

  return 0;
}
