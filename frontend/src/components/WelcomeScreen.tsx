"use client";

import { EXAMPLE_QUESTIONS } from "@/lib/mockData";

interface WelcomeScreenProps {
  onSelectQuestion: (question: string) => void;
}

export default function WelcomeScreen({ onSelectQuestion }: WelcomeScreenProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-12 text-center">
      {/* Animated leaf icon */}
      <div
        className="mb-6 flex h-20 w-20 items-center justify-center rounded-2xl
                   bg-gradient-to-br from-emerald-400 to-green-700 shadow-lg shadow-emerald-500/30
                   animate-[pulse_3s_ease-in-out_infinite]"
        aria-hidden="true"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="white"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
          <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
        </svg>
      </div>

      <h1 className="text-3xl font-bold text-gray-900 sm:text-4xl">
        Welcome to{" "}
        <span className="bg-gradient-to-r from-emerald-600 to-green-500 bg-clip-text text-transparent">
          CropSense AI
        </span>
      </h1>

      <p className="mt-3 max-w-xl text-base text-gray-500 leading-relaxed">
        Your intelligent assistant for crop disease identification, treatment recommendations,
        and agronomic guidance. Ask a question or upload a photo to get started.
      </p>

      {/* Educational disclaimer */}
      <div
        className="mt-5 inline-flex items-center gap-2 rounded-full border border-amber-200
                    bg-amber-50 px-4 py-1.5 text-xs text-amber-700"
        role="status"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        Educational guidance only — not for certified crop diagnosis
      </div>

      {/* Example questions grid */}
      <div className="mt-10 w-full max-w-2xl">
        <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-gray-400">
          Try asking…
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {EXAMPLE_QUESTIONS.map((eq) => (
            <button
              key={eq.id}
              id={`example-question-${eq.id}`}
              onClick={() => onSelectQuestion(eq.question)}
              className={[
                "group relative flex items-start gap-3 rounded-xl border border-gray-200",
                "bg-white px-4 py-3.5 text-left shadow-sm",
                "transition-all duration-200",
                "hover:border-emerald-400 hover:shadow-md hover:shadow-emerald-100",
                "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2",
                "focus-visible:outline-emerald-500",
              ].join(" ")}
            >
              {/* Category badge */}
              <span
                className="mt-0.5 shrink-0 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px]
                            font-semibold text-emerald-700 border border-emerald-200"
              >
                {eq.category}
              </span>

              <div className="min-w-0 flex-1">
                <span className="text-sm font-medium text-gray-800 group-hover:text-emerald-700 transition-colors">
                  {eq.icon} {eq.question}
                </span>
              </div>

              {/* Arrow indicator */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mt-1 shrink-0 text-gray-300 group-hover:text-emerald-500 transition-colors"
                aria-hidden="true"
              >
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
