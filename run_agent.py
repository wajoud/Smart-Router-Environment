import asyncio
import httpx


async def run_routing_agent():
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        print("🤖 AI Agent Connected. Starting Network Optimization...")

        # 1. Reset to get the starting state
        response = await client.post("/reset")
        data = response.json()

        # Handle the observation nesting
        current_obs = data.get("observation", data)

        for i in range(1, 1000):
            # 2. DECIDE: Use the observation from the PREVIOUS step to choose action
            is_congested = current_obs.get("is_congested", False)
            action = 1 if is_congested else 0

            # 3. ACT: Send wrapped action to server
            response = await client.post(
                "/step", json={"action": {"path_selection": action}}
            )
            data = response.json()

            # 4. LEARN: Extract the result of our action
            raw_obs = data.get("observation", {})
            current_obs = (
                raw_obs.get("observation", raw_obs)
                if isinstance(raw_obs, dict)
                else raw_obs
            )

            latency = current_obs.get("latency_ms", 0.0)
            reward = data.get("reward", 0.0)

            # 5. LOG: Show the outcome
            icon = "⚠️" if action == 1 else "✅"
            print(
                f"{icon} Step {i}: Path {'Copper' if action==1 else 'Fiber'} | "
                f"Latency: {latency}ms | Reward: {reward:.2f}"
            )


if __name__ == "__main__":
    asyncio.run(run_routing_agent())
