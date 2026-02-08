//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_GLOBAL_TIMER_H
#define OPENDARTS_FLASH_GLOBAL_TIMER_H
//--------------------------------------------------------------------------

#include <vector>
#include <string>
#include <chrono>

class Timer
{
public:
    enum timer : int { FLASH = 0, STABILITY, SPLIT, EOS, TOTAL };

private:
    std::vector<std::string> timer_names = {"Flash", "Stability", "Split", "EoS"};
    std::vector<std::chrono::time_point<std::chrono::steady_clock>> t0, t1;
    std::vector<double> tot_time;
    std::vector<bool> running;

public:
    Timer();
    ~Timer() = default;

    void start(Timer::timer timer_key);
    void stop(Timer::timer timer_key);
    
    double elapsedMicroseconds(Timer::timer timer_key);
    double elapsedMilliseconds(Timer::timer timer_key) { return this->elapsedMicroseconds(timer_key) * 1e-3; }
    double elapsedSeconds(Timer::timer timer_key) {return this->elapsedMicroseconds(timer_key) * 1e-6; }
    
    // Print timers
	void print_timers();
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_GLOBAL_TIMER_H
//--------------------------------------------------------------------------
