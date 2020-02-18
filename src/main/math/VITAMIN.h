#ifndef VITAMIN_H
#define VITAMIN_H

#include <vector>
#include <queue>

class Neuron {
  public:
    int localIndex;
    std::queue<int> pendingSynapses;
    double isynsAcc{0};
};

namespace vitamin {
	class VITAMINPacket {
		public:
			VITAMINPacket(bool speculative, double time, double xiyi, double Exiyi);
			bool speculative;
			double time;
			double xiyi;
			double Exiyi;

			bool operator <(const VITAMINPacket& packet) const;
	};

	class VITAMINDS {
		public:
			bool haveDataForTime(double t);
			double calculateIsyns(double t, double V);
			void clearSpeculativeDataOlderThan(double t);
			void clearDataOlderThan(double t);
			void addPacket(VITAMINPacket packet); 

		protected:
			double interpolate(double t0, double t1, double y0, double y1, double ti);
			int findPriorPacketIndex(double t); // finds index of the immediately prior packet in the packets vector
			
			std::vector<VITAMINPacket> packets;
	};
}
#endif //VITAMIN_H