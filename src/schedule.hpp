// Copyright (c) 2026 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
// schedule.hpp: Schedule global order of messages on a neuromorphic chip
#ifndef SCHEDULE_HEADER_INCLUDED_
#define SCHEDULE_HEADER_INCLUDED_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional> // For std::reference_wrapper
#include <list>
#include <mutex>
#include <queue>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "chip.hpp"
#include "fwd.hpp"
#include "timestep.hpp"

namespace sanafe
{

enum Direction : uint8_t
{
    // Router link direction
    north = 0U,
    east = 1U,
    south = 2U,
    west = 3U,
    ndirections = 4U,
};

using MessageFifo = std::list<Message>;
using MessagePriorityQueue = std::priority_queue<Message, std::vector<Message>, CompareMessagesBySentTime>;
using TimingModelFunc = std::function<double(TimestepHandle &, Scheduler &, int worker)>;
constexpr size_t max_buffered_timesteps{256UL};

template <typename T, typename Container = std::vector<T>,
        typename Compare = std::less<T>>
class ThreadSafePriorityQueue
{
public:
    ThreadSafePriorityQueue()
            : pq_()
    {
    }

    // Wait until the queue is empty or terminated
    void wait_until_empty()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        empty_cv_.wait(lock, [this] { return pq_.empty() || terminated_; });
    }

    template <typename... Args> void push(Args &&...args)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pq_.push(std::forward<Args>(args)...);
        }
        cv_.notify_one();
    }

    bool pop(T &value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !pq_.empty() || terminated_; });
        if (pq_.empty())
        {
            return false;
        }

        value = pq_.top();
        pq_.pop();
        // If queue is now empty, notify any threads waiting for empty state
        if (pq_.empty())
        {
            empty_cv_.notify_all();
        }

        return true;
    }

    bool try_pop(T &value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pq_.empty())
        {
            return false;
        }
        value = pq_.top();
        pq_.pop();
        if (pq_.empty())
        {
            empty_cv_.notify_all();
        }

        return true;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return pq_.empty();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return pq_.size();
    }

    void set_terminate()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            terminated_ = true;
        }
        cv_.notify_all();
    }

private:
    std::priority_queue<T, Container, Compare> pq_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable empty_cv_;
    bool terminated_ = false;
};

// For comparing timesteps when maintaining priority queue of timesteps
struct CompareTimesteps
{
    bool operator()(const TimestepHandle &lhs, const TimestepHandle &rhs) const
    {
        return lhs->timestep > rhs->timestep;
    }
};

struct Scheduler
{
    std::vector<std::thread> scheduler_threads;
    std::atomic<bool> should_stop{false};
    ThreadSafePriorityQueue<TimestepHandle, std::vector<TimestepHandle>,
            CompareTimesteps>
            timesteps_to_schedule;
    ThreadSafePriorityQueue<TimestepHandle, std::vector<TimestepHandle>,
            CompareTimesteps>
            timesteps_to_write;
    // One Booksim instance per scheduler thread, and one for the serial path.
    std::vector<BookSimInstance *> booksim_instances;
    // First error a scheduler thread hit; schedule_stop_all_threads rethrows it.
    std::mutex error_mutex;
    std::exception_ptr first_error;

    Scheduler() = default;
    // Releases threads and Booksim instances even if a timestep throws.
    ~Scheduler();
    Scheduler(const Scheduler &copy) = delete;
    Scheduler(Scheduler &&other) = delete;
    Scheduler &operator=(const Scheduler &copy) = delete;
    Scheduler &operator=(Scheduler &&other) = delete;

    TimingModel timing_model{timing_model_detailed};
    std::string noc_topology{"mesh"};
    size_t noc_width_in_tiles{0UL};
    size_t noc_height_in_tiles{0UL};
    size_t buffer_size{0UL};
    size_t core_count{0UL};
    size_t max_cores_per_tile{0UL};
    double timestep_sync_delay{0.0};
};

// NocInfo is used by the scheduler to track the high-level state of the NoC
//  at a given time
class NocInfo
{
public:
    std::vector<MessageFifo> messages_received;
    size_t noc_width_in_tiles;
    size_t noc_height_in_tiles;
    size_t core_count;
    size_t max_cores_per_tile;
    // Message density is the distribution of messages buffered in different
    //  links across the NoC. This vector is flattened row-major order
    //  where indexes are (x, y, links) and the links idx change fastest
    std::vector<double> message_density;
    // The time that cores finish receiving their last message, i.e. the
    //  time at which they become free again to process more messages
    std::vector<double> core_finished_receiving;
    double mean_in_flight_receive_delay{0.0};
    long int messages_in_noc{0L};

    explicit NocInfo(const Scheduler &scheduler);
    [[nodiscard]] size_t idx(const size_t x, const size_t y, const size_t link) const
    {
        const size_t links_per_router = max_cores_per_tile + ndirections;
        return (x * noc_height_in_tiles * links_per_router) + (y * links_per_router) +
                link;
    }

    void update_message_density(const Message &message, bool entering_noc);
    void update_rolling_averages(const Message& message, bool entering_noc);
    [[nodiscard]] double calculate_route_congestion(
            const Message &message) const;

private:
    static std::pair<int, int> get_route_xy_increments(const Message &m) noexcept;
};

MessagePriorityQueue schedule_init_timing_priority(std::vector<MessageFifo> &message_queues_per_core);
void schedule_messages(TimestepHandle &ts, Scheduler &scheduler);
double schedule_messages_timestep_simple(TimestepHandle &ts, Scheduler &scheduler, int worker);
double schedule_messages_timestep_detailed(TimestepHandle &ts, Scheduler &scheduler, int worker);
double schedule_messages_timestep_cycle(TimestepHandle &ts, Scheduler &scheduler, int worker);

void schedule_create_threads(Scheduler &scheduler, int scheduler_thread_count);
void schedule_messages_thread(Scheduler &scheduler, int thread_id);
// Waits for the queued work, stops the threads, and rethrows whatever a
//  thread failed with.
void schedule_stop_all_threads(Scheduler &scheduler);
// Stops the threads and frees the Booksim instances without waiting for
//  queued work and without throwing. Safe to call twice.
void schedule_release_resources(Scheduler &scheduler) noexcept;

std::vector<MessageFifo> schedule_init_message_queues(const Timestep &ts, NocInfo &noc);

void schedule_messages_thread(Scheduler &scheduler, const int thread_id);
void schedule_handle_message(Message &m, const Scheduler &scheduler, NocInfo &noc);
double schedule_push_next_message(std::vector<MessageFifo> &messages_sent_per_core, MessagePriorityQueue &priority, const Message &current_message);
void noc_update_message_tracking(const Message &m, NocInfo &noc, bool entering_noc);
void noc_update_all_tracked_messages(double t, NocInfo &noc);

// Router settings that do not depend on the architecture being modelled.
constexpr std::array<std::string_view, 15> booksim_router_config = {{
        "subnets = 2",
        "use_noc_latency = 0",
        "vc_allocator = islip",
        "sw_allocator = islip",
        "alloc_iters = 1",
        "credit_delay = 0",
        "routing_delay = 0",
        "vc_alloc_delay = 1",
        "sw_alloc_delay = 1",
        "input_speedup = 1",
        "output_speedup = 1",
        "internal_speedup = 1.0",
        "packet_size = 1",
        "injection_rate = 0.0",
        "clock_period = 1e-9"
}};

// Builds a Booksim configuration for the NoC. Throws if Booksim cannot model it.
std::vector<std::string> booksim_config_from_noc(const std::string &topology,
        size_t width_in_tiles, size_t height_in_tiles,
        size_t cores_per_tile, size_t link_buffer_size);

}

#endif