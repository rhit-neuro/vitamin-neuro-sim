#include "VITAMIN.h"
#include <iostream>
#include <mpi.h>

vitamin::VITAMINPacket::VITAMINPacket(bool speculative, double time, double xiyi, double Exiyi)
{
	this->speculative = speculative;
	this->time = time;
	this->xiyi = xiyi;
	this->Exiyi = Exiyi;
}

bool vitamin::VITAMINDS::haveDataForTime(double t)
{
	if (packets.empty())
		return false;

	if (packets.back()->time == t)
		return true;

	if (packets.size() < 2)
		return false;

	if (packets.back()->time < t)
		return false;

	return true;
}

double vitamin::VITAMINDS::calculateIsyns(double t, double V)
{
	int prior = findPriorPacketIndex(t);
	prior = (prior == -1) ? 0 : prior;
	double xiyi, Exiyi;
	xiyi = Exiyi = 0; // BEWARE

	for (unsigned int i=prior; i<packets.size(); i++)
	{
		if (packets[i]->time == t)
		{
			xiyi = packets[i]->xiyi;
			Exiyi = packets[i]->Exiyi;
			break;
		}
		else if (packets[i]->time > t)
		{
			VITAMINPacket* priorPacket = packets[prior];
			VITAMINPacket* afterPacket = packets[i];

			xiyi = interpolate(priorPacket->time, afterPacket->time, priorPacket->xiyi, afterPacket->xiyi, t);
			Exiyi = interpolate(priorPacket->time, afterPacket->time, priorPacket->Exiyi, afterPacket->Exiyi, t);
			//std::cout << "Interpolated results " << xiyi << " " << Exiyi << std::endl;
			break;
		}
	}
	
	return V*xiyi - Exiyi;
}

void vitamin::VITAMINDS::clearSpeculativeDataOlderThan(double t)
{
	auto iter = packets.begin();
	while (iter != packets.end())
	{
		if ((*iter)->time >= t)
			break;
		else if ((*iter)->speculative)
			iter = packets.erase(iter);
		else
			++iter;
	}

	// std::vector<VITAMINPacket*> newPackets;
	// for (unsigned int i=0; i<packets.size(); i++)
	// 	if (packets[i]->speculative && (packets[i]->time < t))
	// 		delete packets[i];
	// 	else
	// 		newPackets.push_back(packets[i]);

	// packets = newPackets;
}

void vitamin::VITAMINDS::clearDataOlderThan(double t)
{
	int prior = findPriorPacketIndex(t);
	prior = (prior == -1) ? 0 : prior;
	packets.erase(packets.begin(), packets.begin()+prior);

	// for (unsigned int i=0; i<packets.size(); i++)
	// {
	// 	if (packets[i]->time > t)
	// 	{
	// 		packets.erase(packets.begin(), packets.begin()+i); //noninclusive endpoint
	// 		return;
	// 	}
	// }
}

void vitamin::VITAMINDS::addPacket(VITAMINPacket* packet)
{
	unsigned int insertIndex = findPriorPacketIndex(packet->time)+1; // vector::insert inserts before the specified position
	// std::cout << "Inserting packet t=" << packet->time << " at index " << insertIndex << std::endl;
	
	if (insertIndex >= packets.size()) // 
		packets.insert(packets.end(), packet);
	else if (packet->time != packets[insertIndex]->time)
		packets.insert(packets.begin() + insertIndex, packet);

	// for (auto const& packet : packets)
	// 	std::cout << packet->time << " ";
	// std::cout << std::endl;
}

double vitamin::VITAMINDS::interpolate(double t0, double t1, double y0, double y1, double ti)
{
	double slope = (y1 - y0) / (t1 - t0);

	//std::cout << t0 << " " << t1 << " " << y0 << " " << y1 << " " << ti << " " << ti*slope+y0 << std::endl;

	return (ti-t0)*slope + y0;
}


int vitamin::VITAMINDS::findPriorPacketIndex(double t)
{
	int left = 0;
	int right = packets.size() - 1;
	int current = left;
	while (left < right)
	{
		current = (left + right) / 2 + 1;
		if (t < packets[current]->time)
			right = current - 1;
		else
			left = current;
	}

	if ((packets.size() != 0) && (packets[current]->time < t))
		return current;
	else
		return current - 1;
}