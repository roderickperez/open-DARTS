#include <numeric>
#include "dartsflash/global/timer.hpp"
#include "dartsflash/global/global.hpp"

Timer::Timer()
{
    this->t0.resize(Timer::timer::TOTAL);
    this->t1.resize(Timer::timer::TOTAL);
    this->tot_time = std::vector<double>(Timer::timer::TOTAL, 0.);
    this->running = std::vector<bool>(Timer::timer::TOTAL, false);
}

void Timer::start(Timer::timer timer_key)
{
    t0[timer_key] = std::chrono::steady_clock::now();
    running[timer_key] = true;
    return;
}

void Timer::stop(Timer::timer timer_key)
{
    t1[timer_key] = std::chrono::steady_clock::now();
    running[timer_key] = false;
    double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1[timer_key] - t0[timer_key]).count();
    tot_time[timer_key] += dt;
    return;
}

double Timer::elapsedMicroseconds(Timer::timer timer_key)
{
    if (running[timer_key])
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0[timer_key]).count();
    }
    else
    {
        return tot_time[timer_key];
    }
}

void Timer::print_timers()
{
    print("Total time", std::accumulate(this->tot_time.begin(), this->tot_time.end(), 0.));
	for (int i = 0; i < Timer::timer::TOTAL; i++)
	{
        Timer::timer timer_key = static_cast<Timer::timer>(i);
		std::cout << timer_names[i] << ": " << this->elapsedMicroseconds(timer_key) << " microseconds\n";
	}
	return;
}
