#include "ProgramConfig.h"
#include <mpi.h>

using namespace config;
using namespace config::offsets;
using namespace std;

void ProgramConfig::loadProtobufConfig(protobuf_config::Config &pc, int* neuronCounts, int** neuronMapping) {
  int mpiRank, mpiSize;
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  protoConfigPtr = &pc;
  auto &s = pc.solver();
  absoluteError = s.abserror();
  relativeError = s.relerror();
  startTime = s.starttime();
  endTime = s.endtime();

  numOfNeurons = neuronCounts[mpiRank];

  for (int i=0; i<pc.synapses_size(); i++) // TODO: Refactor the heck out of this
  {
    const auto &protoSynapse = pc.synapses(i);
    
    bool synapseAtThisRank = false;
    for (int j=0; j<neuronCounts[mpiRank] && !synapseAtThisRank; j++)
      synapseAtThisRank = protoSynapse.source() == neuronMapping[mpiRank][j];
    
    if (synapseAtThisRank)
      synapseIndices.push_back(i);
  }
  numOfSynapses = synapseIndices.size();
  numOfNeuronVariables = numOfNeurons * NUM_OF_NEURON_VARIABLES;
  numOfSynapseVariables = numOfSynapses * NUM_OF_SYNAPSE_VARIABLES;

  initializeNeuronOffsets();
  initializeSynapseOffsets();

  neurons = static_cast<NeuronConstants *>(malloc(numOfNeurons * sizeof(NeuronConstants)));
  if (!neurons) {
    cerr << "Failed to allocate memory for the NeuronConstants array" << "\n";
    exit(1);
  }
  initializeNeuronConstantProperties(neuronMapping);

  synapses = static_cast<SynapseConstants *>(malloc(numOfSynapses * sizeof(SynapseConstants)));
  if (!synapses) {
    cerr << "Failed to allocate memory for the SynapseConstants array" << "\n";
    exit(1);
  }
  initializeSynapseConstantProperties(synapseIndices, neuronCounts, neuronMapping);

  // cout << "Rank " << mpiRank << ": ";
  // for (int i=0; i<neuronCounts[mpiRank]; i++)
  //   cout << "N" << neuronMapping[mpiRank][i] <<", ";
  // cout << endl << "Rank " << mpiRank << ": ";
  // for (int i=0; i<synapseIndices.size(); i++)
  //   cout << "S" << synapseIndices[i] << "->" << synapses[i].destinationRank << ", ";
  // cout << endl;
}

void ProgramConfig::loadProtobufConfig(protobuf_config::Config &pc) {
  protoConfigPtr = &pc;
  auto &s = pc.solver();
  absoluteError = s.abserror();
  relativeError = s.relerror();
  startTime = s.starttime();
  endTime = s.endtime();

  numOfNeurons = pc.neurons_size();
  numOfSynapses = pc.synapses_size();
  numOfNeuronVariables = numOfNeurons * NUM_OF_NEURON_VARIABLES;
  numOfSynapseVariables = numOfSynapses * NUM_OF_SYNAPSE_VARIABLES;

  initializeNeuronOffsets();
  initializeSynapseOffsets();

  neurons = static_cast<NeuronConstants *>(malloc(numOfNeurons * sizeof(NeuronConstants)));
  if (!neurons) {
    cerr << "Failed to allocate memory for the NeuronConstants array" << "\n";
    exit(1);
  }
  initializeNeuronConstantProperties();

  synapses = static_cast<SynapseConstants *>(malloc(numOfSynapses * sizeof(SynapseConstants)));
  if (!synapses) {
    cerr << "Failed to allocate memory for the SynapseConstants array" << "\n";
    exit(1);
  }
  initializeSynapseConstantProperties();
}

storage_type & ProgramConfig::getInitialStateValues(int** neuronMapping) {
  initialStateValues = storage_type(static_cast<unsigned long>(numOfNeuronVariables + numOfSynapseVariables));
  initializeNeuronVariables(neuronMapping);
  initializeSynapseVariables(this->synapseIndices);
  return initialStateValues;
}

storage_type & ProgramConfig::getInitialStateValues() {
  initialStateValues = storage_type(static_cast<unsigned long>(numOfNeuronVariables + numOfSynapseVariables));
  initializeNeuronVariables();
  initializeSynapseVariables();
  return initialStateValues;
}

void ProgramConfig::initializeNeuronOffsets() {
  offset_V = OFF_V * numOfNeurons;
  offset_mk2 = OFF_mk2 * numOfNeurons;
  offset_mp = OFF_mp * numOfNeurons;
  offset_mna = OFF_mna * numOfNeurons;
  offset_hna = OFF_hna * numOfNeurons;
  offset_mcaf = OFF_mcaf * numOfNeurons;
  offset_hcaf = OFF_hcaf * numOfNeurons;
  offset_mcas = OFF_mcas * numOfNeurons;
  offset_hcas = OFF_hcas * numOfNeurons;
  offset_mk1 = OFF_mk1 * numOfNeurons;
  offset_hk1 = OFF_hk1 * numOfNeurons;
  offset_mka = OFF_mka * numOfNeurons;
  offset_hka = OFF_hka * numOfNeurons;
  offset_mkf = OFF_mkf * numOfNeurons;
  offset_mh = OFF_mh * numOfNeurons;
}

void ProgramConfig::initializeSynapseOffsets() {
  offset_A = numOfNeuronVariables + OFF_A * numOfSynapses;
  offset_P = numOfNeuronVariables + OFF_P * numOfSynapses;
  offset_M = numOfNeuronVariables + OFF_M * numOfSynapses;
  offset_g = numOfNeuronVariables + OFF_g * numOfSynapses;
  offset_h = numOfNeuronVariables + OFF_h * numOfSynapses;
}

void ProgramConfig::initializeNeuronConstantProperties() {
  protobuf_config::Config &pc = *protoConfigPtr;
  for (int i = 0; i < numOfNeurons; i++) {
    const auto &protoNeuron = pc.neurons(i);
    NeuronConstants *neuronPtr = neurons + i;
    neuronPtr->gbarna = protoNeuron.gbarna();
    neuronPtr->gbarp = protoNeuron.gbarp();
    neuronPtr->gbarcaf = protoNeuron.gbarcaf();
    neuronPtr->gbarcas = protoNeuron.gbarcas();
    neuronPtr->gbark1 = protoNeuron.gbark1();
    neuronPtr->gbark2 = protoNeuron.gbark2();
    neuronPtr->gbarka = protoNeuron.gbarka();
    neuronPtr->gbarkf = protoNeuron.gbarkf();
    neuronPtr->gbarh = protoNeuron.gbarh();
    neuronPtr->gbarl = protoNeuron.gbarl();
    neuronPtr->ena = protoNeuron.ena();
    neuronPtr->eca = protoNeuron.eca();
    neuronPtr->ek = protoNeuron.ek();
    neuronPtr->eh = protoNeuron.eh();
    neuronPtr->el = protoNeuron.el();
    neuronPtr->capacitance = protoNeuron.capacitance();
    // The incoming array holds global synapse IDs, which is why we're storing the global synapse ID
    // in the SynapseConstants struct now.
    neuronPtr->incoming = const_cast<ProtobufRepeatedInt32 *>(&(protoNeuron.incoming()));
  }
}

void ProgramConfig::initializeNeuronConstantProperties(int** neuronMapping) {
  protobuf_config::Config &pc = *protoConfigPtr;
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  for (int i = 0; i < numOfNeurons; i++) {
    const auto &protoNeuron = pc.neurons(neuronMapping[mpiRank][i]);
    NeuronConstants *neuronPtr = neurons + i;
    neuronPtr->globalID = neuronMapping[mpiRank][i];
    neuronPtr->gbarna = protoNeuron.gbarna();
    neuronPtr->gbarp = protoNeuron.gbarp();
    neuronPtr->gbarcaf = protoNeuron.gbarcaf();
    neuronPtr->gbarcas = protoNeuron.gbarcas();
    neuronPtr->gbark1 = protoNeuron.gbark1();
    neuronPtr->gbark2 = protoNeuron.gbark2();
    neuronPtr->gbarka = protoNeuron.gbarka();
    neuronPtr->gbarkf = protoNeuron.gbarkf();
    neuronPtr->gbarh = protoNeuron.gbarh();
    neuronPtr->gbarl = protoNeuron.gbarl();
    neuronPtr->ena = protoNeuron.ena();
    neuronPtr->eca = protoNeuron.eca();
    neuronPtr->ek = protoNeuron.ek();
    neuronPtr->eh = protoNeuron.eh();
    neuronPtr->el = protoNeuron.el();
    neuronPtr->capacitance = protoNeuron.capacitance();
    // The incoming array holds global synapse IDs, which is why we're storing the global synapse ID
    // in the SynapseConstants struct now.
    neuronPtr->incoming = const_cast<ProtobufRepeatedInt32 *>(&(protoNeuron.incoming()));
  }
}

void ProgramConfig::initializeSynapseConstantProperties(std::vector<int> synapseIndices, int* neuronCounts, int** neuronMapping) {
  protobuf_config::Config &pc = *protoConfigPtr;
  int mpiRank;
  int mpiSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  for (int i = 0; i < synapseIndices.size(); i++) {
    const auto &protoSynapse = pc.synapses(synapseIndices[i]);

    SynapseConstants *synapsePtr = synapses + i;
    synapsePtr->globalID = synapseIndices[i];

    // Now we need to find the true source index for this synapse (we should also save this precalculated value)
    // I would also accept rewriting the whole dang thing so that we don't have to carry all this baggage.
    for (int j=0; j<neuronCounts[mpiRank]; j++)
    {
      if (protoSynapse.source() == neuronMapping[mpiRank][j])
      {
        synapsePtr->source = j; // j is the index into our locally-stored neurons
        break; // and so j is the "source" value we want to be storing for this synapse to find NeuronConstants
      }
    }
    
    synapsePtr->gbarsyng = protoSynapse.gbarsyng();
    synapsePtr->gbarsyns = protoSynapse.gbarsyns();
    synapsePtr->esyn = protoSynapse.esyn();
    synapsePtr->buffering = protoSynapse.buffering();
    synapsePtr->h0 = protoSynapse.h0();
    synapsePtr->thresholdV = protoSynapse.thresholdv();
    synapsePtr->tauDecay = protoSynapse.taudecay();
    synapsePtr->tauRise = protoSynapse.taurise();
    synapsePtr->cGraded = protoSynapse.cgraded();
  }

  for (int rank=0; rank < mpiSize; rank++)
  {
    for (int neuron=0; neuron < neuronCounts[rank]; neuron++)
    {
      const auto &protoNeuron = pc.neurons(neuronMapping[rank][neuron]);
      const auto &incoming = protoNeuron.incoming();
      for (int incomingSynapse=0; incomingSynapse < incoming.size(); incomingSynapse++)
      {
        // C++ is friggin' weird
        std::vector<int>::iterator itr = std::find(synapseIndices.begin(), synapseIndices.end(), incoming[incomingSynapse]);
        if (itr != synapseIndices.cend())
        {
          int index = std::distance(synapseIndices.begin(), itr);
          SynapseConstants *synapsePtr = synapses + index;
          synapsePtr->destinationRank = rank;
          synapsePtr->destID = neuronMapping[rank][neuron];
        }
      }
    }
  }
}

void ProgramConfig::initializeSynapseConstantProperties() {
  protobuf_config::Config &pc = *protoConfigPtr;
  for (int i = 0; i < pc.synapses_size(); i++) {
    const auto &protoSynapse = pc.synapses(i);
    SynapseConstants *synapsePtr = synapses + i;
    synapsePtr->globalID = i;
    synapsePtr->source = protoSynapse.source();
    synapsePtr->gbarsyng = protoSynapse.gbarsyng();
    synapsePtr->gbarsyns = protoSynapse.gbarsyns();
    synapsePtr->esyn = protoSynapse.esyn();
    synapsePtr->buffering = protoSynapse.buffering();
    synapsePtr->h0 = protoSynapse.h0();
    synapsePtr->thresholdV = protoSynapse.thresholdv();
    synapsePtr->tauDecay = protoSynapse.taudecay();
    synapsePtr->tauRise = protoSynapse.taurise();
    synapsePtr->cGraded = protoSynapse.cgraded();
  }
}

void ProgramConfig::initializeNeuronVariables(int** neuronMapping) {
  protobuf_config::Config &pc = *protoConfigPtr;
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  for (int i = 0; i < numOfNeurons; i++) {
    const auto &protoNeuron = pc.neurons(neuronMapping[mpiRank][i]);
    initialStateValues[offset_V + i] = protoNeuron.ivoltage();
    initialStateValues[offset_mk2 + i] = protoNeuron.imk2();
    initialStateValues[offset_mp + i] = protoNeuron.imp();
    initialStateValues[offset_mna + i] = protoNeuron.imna();
    initialStateValues[offset_hna + i] = protoNeuron.ihna();
    initialStateValues[offset_mcaf + i] = protoNeuron.imcaf();
    initialStateValues[offset_hcaf + i] = protoNeuron.ihcaf();
    initialStateValues[offset_mcas + i] = protoNeuron.imcas();
    initialStateValues[offset_hcas + i] = protoNeuron.ihcas();
    initialStateValues[offset_mk1 + i] = protoNeuron.imk1();
    initialStateValues[offset_hk1 + i] = protoNeuron.ihk1();
    initialStateValues[offset_mka + i] = protoNeuron.imka();
    initialStateValues[offset_hka + i] = protoNeuron.ihka();
    initialStateValues[offset_mkf + i] = protoNeuron.imkf();
    initialStateValues[offset_mh + i] = protoNeuron.imh();
  }
}

void ProgramConfig::initializeNeuronVariables() {
  protobuf_config::Config &pc = *protoConfigPtr;
  for (int i = 0; i < numOfNeurons; i++) {
    const auto &protoNeuron = pc.neurons(i);
    initialStateValues[offset_V + i] = protoNeuron.ivoltage();
    initialStateValues[offset_mk2 + i] = protoNeuron.imk2();
    initialStateValues[offset_mp + i] = protoNeuron.imp();
    initialStateValues[offset_mna + i] = protoNeuron.imna();
    initialStateValues[offset_hna + i] = protoNeuron.ihna();
    initialStateValues[offset_mcaf + i] = protoNeuron.imcaf();
    initialStateValues[offset_hcaf + i] = protoNeuron.ihcaf();
    initialStateValues[offset_mcas + i] = protoNeuron.imcas();
    initialStateValues[offset_hcas + i] = protoNeuron.ihcas();
    initialStateValues[offset_mk1 + i] = protoNeuron.imk1();
    initialStateValues[offset_hk1 + i] = protoNeuron.ihk1();
    initialStateValues[offset_mka + i] = protoNeuron.imka();
    initialStateValues[offset_hka + i] = protoNeuron.ihka();
    initialStateValues[offset_mkf + i] = protoNeuron.imkf();
    initialStateValues[offset_mh + i] = protoNeuron.imh();
  }
}

void ProgramConfig::initializeSynapseVariables() {
  protobuf_config::Config &pc = *protoConfigPtr;
  for (int i = 0; i < numOfSynapses; i++) {
    const auto &protoSynapse = pc.synapses(i);
    initialStateValues[offset_A + i] = protoSynapse.ia();
    initialStateValues[offset_P + i] = protoSynapse.ip();
    initialStateValues[offset_M + i] = protoSynapse.im();
    initialStateValues[offset_g + i] = protoSynapse.ig();
    initialStateValues[offset_h + i] = protoSynapse.ih();
  }
}

void ProgramConfig::initializeSynapseVariables(std::vector<int> synapseIndices) {
  protobuf_config::Config &pc = *protoConfigPtr;
  for (int i = 0; i < synapseIndices.size(); i++) {
    const auto &protoSynapse = pc.synapses(synapseIndices[i]);
    initialStateValues[offset_A + i] = protoSynapse.ia();
    initialStateValues[offset_P + i] = protoSynapse.ip();
    initialStateValues[offset_M + i] = protoSynapse.im();
    initialStateValues[offset_g + i] = protoSynapse.ig();
    initialStateValues[offset_h + i] = protoSynapse.ih();
  }
}

