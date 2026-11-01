import { useEffect, useState } from 'react'

type Health = { status: string; model_loaded: boolean; model_version: string }

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then(setHealth)
      .catch(() => setError(true))
  }, [])

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-4 bg-slate-50 px-4 text-slate-800">
      <h1 className="text-4xl font-bold">OralSight</h1>
      <p className="text-slate-500">Hello world — oral lesion screening prototype</p>
      <p className="text-sm">
        Backend:{' '}
        {error ? (
          <span className="text-red-600">unreachable</span>
        ) : health ? (
          <span className="text-green-600">
            {health.status} (model: {health.model_version})
          </span>
        ) : (
          <span className="text-slate-400">checking…</span>
        )}
      </p>
    </main>
  )
}

export default App
