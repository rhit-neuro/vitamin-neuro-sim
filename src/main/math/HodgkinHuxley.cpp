#include "ODE.h"
#include "VITAMIN.h"
#include <mpi.h>

using namespace config;

void ode::hodgkinhuxley::calculateNextState(const storage_type &xs, storage_type &dxdts, double t) {
  static HodgkinHuxleyEquation equationInstance;
  return equationInstance.calculateNextState(xs, dxdts, t);
}

ode::hodgkinhuxley::HodgkinHuxleyEquation::HodgkinHuxleyEquation() {
  this->pc = &(ProgramConfig::getInstance());
  ProgramConfig& c = *pc;

  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  //FIXME - Memory leak and a hack
  this->sendData = (double**) malloc(c.numOfSynapses*sizeof(double*)); 
  for (int i=0; i<c.numOfSynapses; ++i)
    this->sendData[i] = (double*) malloc(4*sizeof(double));

  this->sendRequests = (MPI_Request *) malloc(c.numOfSynapses * sizeof(MPI_Request));
  for (int i=0; i<c.numOfSynapses; ++i)
    this->sendRequests[i] = MPI_REQUEST_NULL;

  for (int i=0; i<c.numOfNeurons; ++i)
  {
    const auto &n = c.getNeuronConstantAt(i);
    std::map<int, vitamin::VITAMINDS> synapseMap;
    this->vitamins.push_back(synapseMap);
  }

  busyWaiters.reserve(c.numOfNeurons);
  for (int i=0; i<c.numOfNeurons; ++i)
    busyWaiters.push_back(0);

  for (int i=0; i<c.numOfNeurons; ++i)
  {
    Neuron newNeuron;
    newNeuron.localIndex = i;
    const auto &n = c.getNeuronConstantAt(i);
    for (int j=0; j<n.incoming->size(); ++j)
    {
      const int globalSynapseIndex = n.incoming->Get(j);
      newNeuron.pendingSynapses.push(globalSynapseIndex);
    }
    this->preBuiltNeuronQueue.push(newNeuron);
  }
}

// Out-of-class initialization is necessary for static variables
bool ode::hodgkinhuxley::HodgkinHuxleyEquation::runIsSpeculative = false;
std::vector<unsigned long long> ode::hodgkinhuxley::HodgkinHuxleyEquation::busyWaiters;

void ode::hodgkinhuxley::HodgkinHuxleyEquation::setSpeculative(bool speculative)
{
  runIsSpeculative = speculative;
}

void ode::hodgkinhuxley::HodgkinHuxleyEquation::broadcastIsynsValues(
      double *arrP, double *arrM, double *arrG, SynapseConstants *allSynapses, double t)
{
  for (int i = 0; i < pc->numOfSynapses; ++i) {
    const SynapseConstants &s = allSynapses[i];
    const double Esyn = s.esyn;
    const double gbarsyng = s.gbarsyng;
    const double gbarsyns = s.gbarsyns;
    const double cGraded = s.cGraded;
    const double tauDecay = s.tauDecay;
    const double tauRise = s.tauRise;
    const double P = arrP[i];
    const double M = arrM[i];
    const double g = arrG[i];
    const double P3 = pow(P, 3);
    double isyng = gbarsyng * P3 / (cGraded + P3);
    const double t0 = 0;
    const double tPeak = t0 + (tauDecay * tauRise * log(tauDecay/tauRise)) / (tauDecay - tauRise);
    const double fsyns = 1 / (exp(-(tPeak - t0)/tauDecay) + exp(-(tPeak - t0)/tauRise));
    const double isyns = M * gbarsyns * g * fsyns;
    
    double xiyi = isyns + isyng;

    sendData[i][0] = (runIsSpeculative) ? 1.0 : 0.0;
    sendData[i][1] = t;
    sendData[i][2] = xiyi;
    sendData[i][3] = Esyn*xiyi;

    if (s.destinationRank == mpiRank)
    {
      for (int neuron=0; neuron < pc->numOfNeurons; ++neuron)
      {
        const NeuronConstants &n = pc->getNeuronConstantAt(neuron);
        if (n.globalID == s.destID)
        {
          vitamin::VITAMINDS& relevantVitamin = vitamins[neuron][s.globalID];
          vitamin::VITAMINPacket packet(runIsSpeculative, sendData[i][1], sendData[i][2], sendData[i][3]);
          relevantVitamin.addPacket(packet);
          break;
        }    
      }
    }
    else
    {
      MPI_Isend(sendData[i], 4, MPI_DOUBLE, s.destinationRank, s.globalID, MPI_COMM_WORLD, &(sendRequests[i]));
    }
  }
}

void ode::hodgkinhuxley::HodgkinHuxleyEquation::calculateNextState(const storage_type &x, storage_type &dxdt, double t) {
  ProgramConfig &c = *pc;
  double *arrV = c.getVArray(const_cast<storage_type &>(x));
  double *arrMk2 = c.getMk2Array(const_cast<storage_type &>(x));
  double *arrMp = c.getMpArray(const_cast<storage_type &>(x));
  double *arrMna = c.getMnaArray(const_cast<storage_type &>(x));
  double *arrHna = c.getHnaArray(const_cast<storage_type &>(x));
  double *arrMcaf = c.getMcafArray(const_cast<storage_type &>(x));
  double *arrHcaf = c.getHcafArray(const_cast<storage_type &>(x));
  double *arrMcas = c.getMcasArray(const_cast<storage_type &>(x));
  double *arrHcas = c.getHcasArray(const_cast<storage_type &>(x));
  double *arrMk1 = c.getMk1Array(const_cast<storage_type &>(x));
  double *arrHk1 = c.getHk1Array(const_cast<storage_type &>(x));
  double *arrMka = c.getMkaArray(const_cast<storage_type &>(x));
  double *arrHka = c.getHkaArray(const_cast<storage_type &>(x));
  double *arrMkf = c.getMkfArray(const_cast<storage_type &>(x));
  double *arrMh = c.getMhArray(const_cast<storage_type &>(x));
  double *arrA = c.getAArray(const_cast<storage_type &>(x));
  double *arrP = c.getPArray(const_cast<storage_type &>(x));
  double *arrM = c.getMArray(const_cast<storage_type &>(x));
  double *arrG = c.getGArray(const_cast<storage_type &>(x));
  double *arrH = c.getHArray(const_cast<storage_type &>(x));

  double *arrdVdt = c.getVArray(dxdt);
  double *arrdMk2dt = c.getMk2Array(dxdt);
  double *arrdMpdt = c.getMpArray(dxdt);
  double *arrdMnadt = c.getMnaArray(dxdt);
  double *arrdHnadt = c.getHnaArray(dxdt);
  double *arrdMcafdt = c.getMcafArray(dxdt);
  double *arrdHcafdt = c.getHcafArray(dxdt);
  double *arrdMcasdt = c.getMcasArray(dxdt);
  double *arrdHcasdt = c.getHcasArray(dxdt);
  double *arrdMk1dt = c.getMk1Array(dxdt);
  double *arrdHk1dt = c.getHk1Array(dxdt);
  double *arrdMkadt = c.getMkaArray(dxdt);
  double *arrdHkadt = c.getHkaArray(dxdt);
  double *arrdMkfdt = c.getMkfArray(dxdt);
  double *arrdMhdt = c.getMhArray(dxdt);
  double *arrdAdt = c.getAArray(dxdt);
  double *arrdPdt = c.getPArray(dxdt);
  double *arrdMdt = c.getMArray(dxdt);
  double *arrdGdt = c.getGArray(dxdt);
  double *arrdHdt = c.getHArray(dxdt);

  if (!runIsSpeculative)
  {
    for (int i=0; i<vitamins.size(); ++i)
    {
      auto iter = vitamins[i].begin();
      while (iter != vitamins[i].end())
      {
        iter->second.clearDataOlderThan(t-0.03);  //empirically determined to work pretty well
        iter->second.clearSpeculativeDataOlderThan(t);
        ++iter; // WHAT. THE. HECK.
      }
    }
  }
  broadcastIsynsValues(arrP, arrM, arrG, c.getAllSynapseConstants(), t);

  const int numOfNeurons = c.numOfNeurons;
  const int numOfSynapses = c.numOfSynapses;

  // Copy-construction doesn't adversely affect runtime and is easier on the eyes
  std::queue<Neuron> neurons(this->preBuiltNeuronQueue);

  while (!neurons.empty())
  {
    Neuron neuron = neurons.front(); neurons.pop(); // This is a pop operation now
    bool busyWaiting = true;
    for (unsigned int i=0; i<neuron.pendingSynapses.size(); ++i)
    {
      int globalSynapse = neuron.pendingSynapses.front();
      neuron.pendingSynapses.pop();
      vitamin::VITAMINDS& relevantVitamin = vitamins[neuron.localIndex][globalSynapse];

      if (relevantVitamin.haveDataForTime(t))
      {
        busyWaiting = false;
        neuron.isynsAcc += relevantVitamin.calculateIsyns(t, arrV[neuron.localIndex]);
        continue;
      }     

      int flag = 0;
      MPI_Status status;
      MPI_Iprobe(MPI_ANY_SOURCE, globalSynapse, MPI_COMM_WORLD, &flag, &status);

      if (flag)
      {
        busyWaiting = false;
        double recvData[4];
        MPI_Recv(recvData, 4, MPI_DOUBLE, status.MPI_SOURCE, globalSynapse, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vitamin::VITAMINPacket packet((recvData[0] == 1), recvData[1], recvData[2], recvData[3]);
        relevantVitamin.addPacket(packet);
      }

      if (relevantVitamin.haveDataForTime(t))
      {
        busyWaiting = false;
        neuron.isynsAcc += relevantVitamin.calculateIsyns(t, arrV[neuron.localIndex]);
      }
      else // put this synapse back in the queue
      {
        neuron.pendingSynapses.push(globalSynapse);
      }
    }

    if (neuron.pendingSynapses.empty())
    {
      busyWaiting = false;
      int i = neuron.localIndex;
      const NeuronConstants &n = c.getNeuronConstantAt(i);
      const double V = arrV[i];

      // Calculate dVdt
      arrdVdt[i] = -(ina(n.gbarna, arrMna[i], arrHna[i], V, n.ena) +
                     ip(n.gbarp, arrMp[i], V, n.ena) +
                     icaf(n.gbarcaf, arrMcaf[i], arrHcaf[i], V, n.eca) +
                     icas(n.gbarcas, arrMcas[i], arrHcas[i], V, n.eca) +
                     ik1(n.gbark1, arrMk1[i], arrHk1[i], V, n.ek) +
                     ik2(n.gbark2, arrMk2[i], V, n.ek) +
                     ika(n.gbarka, arrMka[i], arrHka[i], V, n.ek) +
                     ikf(n.gbarkf, arrMkf[i], V, n.ek) +
                     ih(n.gbarh, arrMh[i], V, n.eh) +
                     il(n.gbarl, V, n.el) +
                     neuron.isynsAcc
      ) / n.capacitance;

      // Calculate dMk2dt
      arrdMk2dt[i] = dMk2dt(V, arrMk2[i]);
      // Calculate dMpdt
      arrdMpdt[i] = dMpdt(V, arrMp[i]);
      // Calculate dMnadt
      arrdMnadt[i] = dMnadt(V, arrMna[i]);
      // Calculate dHnadt
      arrdHnadt[i] = dHnadt(V, arrHna[i]);
      // Calculate dMcafdt
      arrdMcafdt[i] = dMcafdt(V, arrMcaf[i]);
      // Calculate dHcafdt
      arrdHcafdt[i] = dHcafdt(V, arrHcaf[i]);
      // Calculate dMcasdt
      arrdMcasdt[i] = dMcasdt(V, arrMcas[i]);
      // Calculate dHcasdt
      arrdHcasdt[i] = dHcasdt(V, arrHcas[i]);
      // Calculate dMk1dt
      arrdMk1dt[i] = dMk1dt(V, arrMk1[i]);
      // Calculate dHk1dt
      arrdHk1dt[i] = dHk1dt(V, arrHk1[i]);
      // Calculate dMkadt
      arrdMkadt[i] = dMkadt(V, arrMka[i]);
      // Calculate dHkadt
      arrdHkadt[i] = dHkadt(V, arrHka[i]);
      // Calculate dMkfdt
      arrdMkfdt[i] = dMkfdt(V, arrMkf[i]);
      // Calculate dMhdt
      arrdMhdt[i] = dMhdt(V, arrMh[i]);
    }
    else // This neuron still has work to be done, so put it back on the queue
    {
      neurons.push(neuron);
    }

    if (busyWaiting)
      ++busyWaiters[neuron.localIndex];
  }

#if USE_OPENMP
  #pragma omp parallel for default(shared)
#endif
  for (int j = 0; j < numOfSynapses; ++j) {
    using namespace std;
    const SynapseConstants &s = c.getSynapseConstantAt(j);
    const int sourceNeuronIndex = s.source;
    const NeuronConstants &n = c.getNeuronConstantAt(0);
    const double V = arrV[sourceNeuronIndex];

    // Calculate dPdt
    // TODO Cache the result from loop above to save time
    arrdPdt[j] = ica(
      icaf(n.gbarcaf, arrMcaf[sourceNeuronIndex], arrHcaf[sourceNeuronIndex], V, n.eca),
      icas(n.gbarcas, arrMcas[sourceNeuronIndex], arrHcas[sourceNeuronIndex], V, n.eca),
      arrA[j]
    ) - s.buffering * arrP[j];

    // Calculate dAdt
    arrdAdt[j] = (1.0e-10 / (1 + exp(-100.0 * (V + 0.02))) - arrA[j]) / 0.2;
    // Calculate dMdt
    arrdMdt[j] = (0.1 + 0.9 / (1 + exp(-1000.0 * (V + 0.04))) - arrM[j]) / 0.2;
    // Calculate dGdt
    arrdGdt[j] = -arrG[j] / s.tauDecay + arrH[j];
    // Calculate dHdt
    arrdHdt[j] = -arrH[j] / s.tauRise + (V > s.thresholdV ? s.h0 : 0);
  }

  MPI_Waitall(numOfSynapses, sendRequests, MPI_STATUSES_IGNORE);
  setSpeculative(true);
}
