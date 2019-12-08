#include "ODE.h"
#include <mpi.h>

using namespace config;

void ode::hodgkinhuxley::calculateNextState(const storage_type &xs, storage_type &dxdts, double t) {
  static HodgkinHuxleyEquation equationInstance;
  return equationInstance.calculateNextState(xs, dxdts, t);
}

ode::hodgkinhuxley::HodgkinHuxleyEquation::HodgkinHuxleyEquation() {
  this->pc = &(ProgramConfig::getInstance());

  int numOfNeurons = pc->numOfNeurons;
  // // FIXME: MPI_Alloc rather than malloc for now; may be buggy as all get-out
  // malloc(numOfNeurons*sizeof(double), MPI_INFO_NULL, &isynsXY);
  // malloc(numOfNeurons*sizeof(double), MPI_INFO_NULL, &isynsEXY);
  //isynsXY = (double*) malloc(numOfNeurons*sizeof(double));
  //isynsEXY = (double*) malloc(numOfNeurons*sizeof(double));

  //MPI_Win_create(isynsXY, numOfNeurons*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winXY);
  //MPI_Win_create(isynsEXY, numOfNeurons*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winEXY); 
}
//FIXME - We aren't freeing these currently because putting that in a destructor gets called after MPI_Finalize
// and throws a nasty error. This should be fixed in the long-term but we need to do a bigger refactor anyway.


//FIXME - this should iterate over the neurons in this rank and work on each of their synapses
// This will work because this rank is responsible for every synapse leaving one of its neurons
void isyns_pt1(double *arrP, double *arrM, double *arrG,
                                          SynapseConstants *allSynapses, MPI_Win &winXY, MPI_Win &winEXY)
                                         // ProtobufRepeatedInt32 &ownSynapses, int numOfOwnSynapses) 
{
  //FIXME: I don't like copy-pasting this crap all over the place
  int mpiRank, mpiSize;
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  //FIXME: THIS IS A HACK!!!!
  int numOfOwnSynapses = 1;
  int synapseIndex = 0;

  // TODO Investigate the formula expressed here. It's not the same as in the paper
  //#pragma omp parallel for reduction(+:result)
  for (int i = 0; i < numOfOwnSynapses; i++) {
    //const int synapseIndex = ownSynapses[i];
    const SynapseConstants &s = allSynapses[i]; //allSynapses[synapseIndex];
    const double Esyn = s.esyn;
    const double gbarsyng = s.gbarsyng;
    const double gbarsyns = s.gbarsyns;
    const double cGraded = s.cGraded;
    const double tauDecay = s.tauDecay;
    const double tauRise = s.tauRise;
    const double P = arrP[synapseIndex];
    const double M = arrM[synapseIndex];
    const double g = arrG[synapseIndex];
    const double P3 = pow(P, 3);
    double isyng = gbarsyng * P3 / (cGraded + P3);
    // TODO Investigate: magic number t0
    const double t0 = 0;
    const double tPeak = t0 + (tauDecay * tauRise * log(tauDecay/tauRise)) / (tauDecay - tauRise);
    // TODO Investigate: why
    const double fsyns = 1 / (exp(-(tPeak - t0)/tauDecay) + exp(-(tPeak - t0)/tauRise));
    const double isyns = M * gbarsyns * g * fsyns;
    
    double xiyi = isyns + isyng;
    double esynXiYi = Esyn*xiyi;

    if (mpiRank == 0)
    { 
      MPI_Accumulate(&xiyi, 1, MPI_DOUBLE, 1, 0, 1, MPI_DOUBLE, MPI_SUM, winXY);
      MPI_Accumulate(&esynXiYi, 1, MPI_DOUBLE, 1, 0, 1, MPI_DOUBLE, MPI_SUM, winEXY);
    }
    else
    {
      MPI_Accumulate(&xiyi, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, winXY);
      MPI_Accumulate(&esynXiYi, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, winEXY);
    }
  }
}

void ode::hodgkinhuxley::HodgkinHuxleyEquation::calculateNextState(const storage_type &x, storage_type &dxdt, double t) {
  //FIXME: I don't like copy-pasting this crap all over the place
  int mpiRank, mpiSize;
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

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

  const int numOfNeurons = c.numOfNeurons;
  const int numOfSynapses = c.numOfSynapses;

#if USE_OPENMP
  #pragma omp parallel for default(shared)
#endif
  for (int i = 0; i < numOfNeurons; i++) {
    using namespace std;
    const NeuronConstants &n = c.getNeuronConstantAt(i);
    const double V = arrV[i];

    //std::cout << mpiRank << " has xy=" << isynsXY[i] << " and exy=" << isynsEXY[1] << endl;

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
                   isyns(V, arrP, arrM, arrG, c.getAllSynapseConstants(), *(n.incoming), n.incoming->size())
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
  
#if USE_OPENMP
  #pragma omp parallel for default(shared)
#endif
  for (int j = 0; j < numOfSynapses; j++) {
    using namespace std;
    const SynapseConstants &s = c.getSynapseConstantAt(j);
    // FIXME!! - This needs to be translated to our rank's index of neurons.
    const int sourceNeuronIndex = 0; //s.source;
    const NeuronConstants &n = c.getNeuronConstantAt(sourceNeuronIndex);
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
}
