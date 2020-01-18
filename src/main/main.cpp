#include <iostream>

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
  for (int i = 0; i < bufferSize-1; i++) {
    toWrite[i+1] = x[i];
  }
  buffer->writeData(&(toWrite[0]));
}

int main(int argc, char** argv) {
  TimeLogger &tLogger = TimeLogger::getInstance();
  tLogger.recordProgramStartTime();

  po::variables_map vm;
  if (!argparser::parse(argc, argv, vm)) {
    return 0;
  }

  const auto &inputFile = vm["input-file"].as<string>();
  const auto &outputFile = vm["output-file"].as<string>();

  tLogger.recordLoadConfigStartTime();
  JsonToProtobufConfigConverter converter;
  Config config = converter.readConfig(const_cast<string &>(inputFile));
  config::ProgramConfig &c = config::ProgramConfig::getInstance();
  try {
    c.loadProtobufConfig(config);
  } catch (exception &e) {
    cerr << e.what() << endl;
    return 1;
  }
  tLogger.recordLoadConfigEndTime();

  const auto numNeuron = config.neurons_size();
  const int bufferSize = 1 + numNeuron;
  //const int bufferSize = 1+(15 + 5)*2;  // time plus neuron vals plus synapse vals --> no derivatives
  const int precision = vm["output-precision"].as<int>();
  const int verbosity = vm["verbose-level"].as<int>();
  auto buffer = new AsyncBuffer(bufferSize, const_cast<string &>(outputFile), precision, verbosity);

  sequential::ode_system_function *equation = factory::equation::getEquation(vm);

  tLogger.recordCalculationStartTime();

  storage_type x = c.getInitialStateValues();
  auto myStepper = make_controlled(c.absoluteError, c.relativeError, runge_kutta_dopri5<storage_type>());
  auto timeData = (double*) malloc(2*sizeof(double));
  double observerStep = 0.00025;
  timeData[0] = c.startTime;
  timeData[1] = observerStep;

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

  tLogger.recordCalculationEndTime();

  delete buffer;
  tLogger.recordProgramEndTime();
  tLogger.printSummary();

  return 0;
}
