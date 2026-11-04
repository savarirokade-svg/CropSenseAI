// ---------------------------------------------------------------------------
// Mock data – sample conversations and example questions
// ---------------------------------------------------------------------------

import { Conversation } from "@/types/chat";

/** Pre-populated sidebar conversation history */
export const SAMPLE_CONVERSATIONS: Conversation[] = [
  {
    id: "conv-1",
    title: "Tomato early blight symptoms",
    updatedAt: "2026-10-13T10:00:00Z",
  },
  {
    id: "conv-2",
    title: "Potato late blight management",
    updatedAt: "2026-10-12T14:30:00Z",
  },
  {
    id: "conv-3",
    title: "Tomato Septoria leaf spot",
    updatedAt: "2026-10-11T09:15:00Z",
  },
  {
    id: "conv-4",
    title: "Potato common scab prevention",
    updatedAt: "2026-10-10T16:45:00Z",
  },
  {
    id: "conv-5",
    title: "Tomato Fusarium wilt risks",
    updatedAt: "2026-10-09T11:20:00Z",
  },
];

/**
 * Clickable example questions shown on the welcome screen.
 * Limited to the crops the knowledge base actually covers (tomato, potato).
 */
export const EXAMPLE_QUESTIONS = [
  {
    id: "eq-1",
    icon: "🍅",
    question: "What are the early symptoms of tomato early blight?",
    category: "Disease ID",
  },
  {
    id: "eq-2",
    icon: "🥔",
    question: "How do I tell tomato late blight apart from potato late blight?",
    category: "Identification",
  },
  {
    id: "eq-3",
    icon: "🍅",
    question: "What are the symptoms of Septoria leaf spot in tomato?",
    category: "Identification",
  },
  {
    id: "eq-4",
    icon: "🥔",
    question: "How is potato common scab managed?",
    category: "Treatment",
  },
  {
    id: "eq-5",
    icon: "🔬",
    question: "Can I upload a photo for crop disease analysis?",
    category: "Image Analysis",
  },
  {
    id: "eq-6",
    icon: "🌱",
    question: "What conditions favour tomato Fusarium wilt?",
    category: "Risk Factors",
  },
];
