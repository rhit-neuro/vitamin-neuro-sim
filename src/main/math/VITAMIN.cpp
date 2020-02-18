#include "VITAMIN.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>



vitamin::VITAMINPacket::VITAMINPacket(bool speculative, double time, double xiyi, double Exiyi)
{
	this->speculative = speculative;
	this->time = time;
	this->xiyi = xiyi;
	this->Exiyi = Exiyi;
}

bool vitamin::VITAMINPacket::operator <(const VITAMINPacket& packet) const
{
	return time < packet.time;
}

bool vitamin::VITAMINDS::haveDataForTime(double t)
{
	if (packets.empty())
		return false;

	if (packets.back().time == t)
		return true;

	if (packets.size() < 2)
		return false;

	if (packets.back().time < t)
		return false;

	return true;
}

double vitamin::VITAMINDS::calculateIsyns(double t, double V)
{
	int prior = findPriorPacketIndex(t);
	double xiyi, Exiyi;
	xiyi = Exiyi = 0; // BEWARE

	// if (prior > 0)
	// 	--prior;

	// std::cout << "Calculating at t=" << t << std::endl;

	VITAMINPacket priorPacket(true, -1, 0, 0);

	for (unsigned int i=prior; i<packets.size(); ++i)
	{
		if (packets[i].time == t)
		{
			xiyi = packets[i].xiyi;
			Exiyi = packets[i].Exiyi;
			break;
		}
		else if (packets[i].time > t)
		{
			VITAMINPacket afterPacket = packets[i];
			// if (priorPacket.time == -1)
			// {
			// 	std::cout << "Breaking at index " << i << ", prior " << prior << std::endl;
			// 	MPI_Abort(MPI_COMM_WORLD, 1);
			// }

			xiyi = interpolate(priorPacket.time, afterPacket.time, priorPacket.xiyi, afterPacket.xiyi, t);
			Exiyi = interpolate(priorPacket.time, afterPacket.time, priorPacket.Exiyi, afterPacket.Exiyi, t);

			//std::cout << "Interpolated results " << xiyi << " " << Exiyi << std::endl;
			break;
		}

		if (packets[i].time > priorPacket.time)
		{
			priorPacket = packets[i];
		}

	}
	
	return V*xiyi - Exiyi;
}

void vitamin::VITAMINDS::clearSpeculativeDataOlderThan(double t)
{
	packets.erase(std::remove_if(packets.begin(), packets.end(),
								[&](VITAMINPacket packet) { return (packet.time < t) && (packet.speculative); }),
					packets.end());
}

void vitamin::VITAMINDS::clearDataOlderThan(double t)
{
	int prior = findPriorPacketIndex(t);
	packets.erase(packets.begin(), packets.begin()+prior);
}

void vitamin::VITAMINDS::addPacket(VITAMINPacket packet)
{
	packets.insert(std::lower_bound(packets.begin(), packets.end(), packet), packet);

	// int insertIndex = std::upper_bound(packets.begin(), packets.end(), packet) - packets.begin();
	// std::cout << "Inserting packet t=" << packet.time << " at index " << insertIndex << std::endl;

	// for (auto const& packet : packets)
	// 	std::cout << packet.time << " ";
	// std::cout << std::endl;
}

double vitamin::VITAMINDS::interpolate(double t0, double t1, double y0, double y1, double ti)
{
	// http://paulbourke.net/miscellaneous/interpolation/

	// -- Linear Interpolation -- //
	double slope = (y1 - y0) / (t1 - t0);
	return (ti-t0)*slope + y0;

	// // -- Cosine Interpolation -- //
	// double mu = (ti-t0) /(t1-t0);
	// double mu2 = (1-cos(mu*M_PI))/2;
	// return y0*(1-mu2) + y1*mu2;
}


int vitamin::VITAMINDS::findPriorPacketIndex(double t)
{
	VITAMINPacket packet(true, t, 0, 0);
	// an iterator pointing to the first index in packets where Packet.time < t.
	auto lower = std::lower_bound(packets.begin(), packets.end(), packet);

	return lower - packets.begin();
}