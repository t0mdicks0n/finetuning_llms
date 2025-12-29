import { useState, useEffect, type FormEvent } from 'react'

// Modal API endpoint - update this after deployment
const MODAL_API_URL = import.meta.env.VITE_MODAL_API_URL || ''

// Warmup question (short, simple)
const WARMUP_QUESTION = 'Hej'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatResponse {
  response: string
  model: string
  error?: string
}

interface ComparisonResult {
  question: string
  vanilla: ChatResponse | null
  finetuned: ChatResponse | null
  loading: boolean
}

type WarmupStatus = 'idle' | 'warming' | 'ready' | 'error'

async function fetchResponse(messages: Message[], useFinetuned: boolean): Promise<ChatResponse> {
  if (!MODAL_API_URL) {
    return { response: '', model: '', error: 'API URL not configured. Set VITE_MODAL_API_URL.' }
  }

  const response = await fetch(MODAL_API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      use_finetuned: useFinetuned,
      max_tokens: 512,
      temperature: 0.7,
    }),
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`)
  }

  return response.json()
}

function WarmupScreen({ status, onSkip }: { status: WarmupStatus; onSkip: () => void }) {
  const messages: Record<WarmupStatus, string> = {
    idle: 'Förbereder...',
    warming: 'Startar GPU-servrar...',
    ready: 'Redo!',
    error: 'Kunde inte ansluta till servern',
  }

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="text-center px-4">
        <h1 className="text-3xl font-bold text-white mb-2">Swedish Sovereign AI</h1>
        <p className="text-gray-400 mb-8">Jämför vanilla Mistral-7B med Riksbanken-finjusterade modell</p>

        <div className="mb-8">
          {status === 'warming' && (
            <div className="flex flex-col items-center gap-4">
              <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full animate-pulse" style={{ width: '60%' }}></div>
              </div>
              <p className="text-gray-300">{messages[status]}</p>
              <p className="text-gray-500 text-sm">Detta kan ta upp till 2 minuter första gången...</p>
            </div>
          )}
          {status === 'error' && (
            <div className="flex flex-col items-center gap-4">
              <p className="text-red-400">{messages[status]}</p>
              <button
                onClick={onSkip}
                className="text-blue-400 hover:text-blue-300 underline"
              >
                Försök ändå
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ResponseCard({
  title,
  response,
  loading,
  accentColor
}: {
  title: string
  response: ChatResponse | null
  loading: boolean
  accentColor: string
}) {
  return (
    <div className={`flex-1 bg-gray-800 rounded-lg border-t-4 ${accentColor} p-4`}>
      <h3 className="text-lg font-semibold text-gray-200 mb-3">{title}</h3>
      <div className="min-h-[200px]">
        {loading ? (
          <div className="flex items-center justify-center h-[200px]">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-400"></div>
          </div>
        ) : response?.error ? (
          <p className="text-red-400">{response.error}</p>
        ) : response ? (
          <div>
            <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">{response.response}</p>
            <p className="text-xs text-gray-500 mt-4">Modell: {response.model}</p>
          </div>
        ) : (
          <p className="text-gray-500 italic">Svaret visas här...</p>
        )}
      </div>
    </div>
  )
}

function App() {
  const [input, setInput] = useState('')
  const [comparisons, setComparisons] = useState<ComparisonResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [warmupStatus, setWarmupStatus] = useState<WarmupStatus>('idle')

  // Warmup on mount
  useEffect(() => {
    const warmup = async () => {
      setWarmupStatus('warming')

      try {
        // Send a simple warmup request to both models in parallel
        const warmupMessages: Message[] = [{ role: 'user', content: WARMUP_QUESTION }]
        await Promise.all([
          fetchResponse(warmupMessages, false),
          fetchResponse(warmupMessages, true),
        ])
        setWarmupStatus('ready')
      } catch (error) {
        console.error('Warmup failed:', error)
        setWarmupStatus('error')
      }
    }

    if (MODAL_API_URL) {
      warmup()
    } else {
      setWarmupStatus('error')
    }
  }, [])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const question = input.trim()
    setInput('')
    setIsLoading(true)

    // Add new comparison with loading state
    const newComparison: ComparisonResult = {
      question,
      vanilla: null,
      finetuned: null,
      loading: true,
    }
    setComparisons(prev => [newComparison, ...prev])

    const messages: Message[] = [{ role: 'user', content: question }]

    // Fetch both responses in parallel
    try {
      const [vanillaResult, finetunedResult] = await Promise.allSettled([
        fetchResponse(messages, false),
        fetchResponse(messages, true),
      ])

      setComparisons(prev => {
        const updated = [...prev]
        updated[0] = {
          ...updated[0],
          vanilla: vanillaResult.status === 'fulfilled'
            ? vanillaResult.value
            : { response: '', model: '', error: (vanillaResult.reason as Error).message },
          finetuned: finetunedResult.status === 'fulfilled'
            ? finetunedResult.value
            : { response: '', model: '', error: (finetunedResult.reason as Error).message },
          loading: false,
        }
        return updated
      })
    } catch (error) {
      console.error('Error fetching responses:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // Show warmup screen until ready
  if (warmupStatus !== 'ready') {
    return <WarmupScreen status={warmupStatus} onSkip={() => setWarmupStatus('ready')} />
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 py-6">
        <div className="max-w-6xl mx-auto px-4">
          <h1 className="text-2xl font-bold text-center">
            Swedish Sovereign AI Demo
          </h1>
          <p className="text-gray-400 text-center mt-2">
            Jämför vanilla Mistral-7B med Riksbanken-finjusterade modell
          </p>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Input form */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ställ en fråga om svensk penningpolitik..."
              className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-medium transition-colors"
            >
              {isLoading ? 'Frågar...' : 'Fråga båda'}
            </button>
          </div>
          <div className="flex flex-wrap gap-2 mt-3">
            <span className="text-gray-500 text-sm">Testa:</span>
            {[
              'Vad är reporäntan?',
              'Vad innebär Riksbankens köp av statsobligationer?',
              'Hur påverkar räntan inflationen?',
              'Vad var Riksbankens största utmaningar 2024?',
              'Vad är KPIF?',
            ].map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => setInput(q)}
                className="text-blue-400 hover:text-blue-300 text-sm hover:underline"
              >
                {q}
              </button>
            ))}
          </div>
        </form>

        {/* Comparisons */}
        <div className="space-y-8">
          {comparisons.map((comparison, index) => (
            <div key={index} className="bg-gray-850 rounded-xl p-6 border border-gray-700">
              {/* Question */}
              <div className="mb-4 pb-4 border-b border-gray-700">
                <span className="text-gray-400 text-sm">Fråga:</span>
                <p className="text-lg text-white font-medium">{comparison.question}</p>
              </div>

              {/* Side by side responses */}
              <div className="flex gap-4 flex-col md:flex-row">
                <ResponseCard
                  title="Vanilla Mistral-7B"
                  response={comparison.vanilla}
                  loading={comparison.loading}
                  accentColor="border-gray-500"
                />
                <ResponseCard
                  title="Riksbanken Fine-tuned"
                  response={comparison.finetuned}
                  loading={comparison.loading}
                  accentColor="border-blue-500"
                />
              </div>
            </div>
          ))}
        </div>

        {/* Empty state */}
        {comparisons.length === 0 && (
          <div className="text-center py-16">
            <p className="text-gray-500 text-lg">
              Ställ en fråga för att se hur modellerna jämför sig
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-700 py-6 mt-auto">
        <p className="text-center text-gray-500 text-sm">
          Finjusterad på Riksbankens penningpolitiska rapporter (2022-2025)
        </p>
      </footer>
    </div>
  )
}

export default App
