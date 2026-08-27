import { createFileRoute } from "@tanstack/react-router";
import {
  Database,
  BarChart3,
  TrendingUp,
  Users,
  FileCode2,
  Github,
  Sparkles,
  LineChart,
  PieChart,
  Table2,
} from "lucide-react";

const DESCRIPTION =
  "A Python-based data science project analyzing a personal loan portfolio with Pandas, NumPy, Matplotlib and Seaborn — EDA, visualizations and business insights.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Personal Loan Portfolio Analysis — Python Data Science Project" },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: "Personal Loan Portfolio Analysis" },
      { property: "og:description", content: DESCRIPTION },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Index,
});

const pipeline = [
  {
    icon: Database,
    title: "Load & Inspect",
    desc: "Reads data/Bank_Personal_Loan_Modelling.csv, reports shape, dtypes, missing values and duplicates.",
  },
  {
    icon: Sparkles,
    title: "Clean",
    desc: "Standardizes column names, fixes negative experience values and converts ID columns to categorical.",
  },
  {
    icon: BarChart3,
    title: "Analyze",
    desc: "Descriptive statistics plus age, income, education, family, mortgage and acceptance-rate analysis.",
  },
  {
    icon: TrendingUp,
    title: "Visualize",
    desc: "Histograms, boxplots, count plots, correlation heatmap and scatter plots saved to outputs/charts/.",
  },
];

const stats = [
  { icon: Users, label: "Customers", value: "5,000" },
  { icon: Table2, label: "Features", value: "14" },
  { icon: PieChart, label: "Charts", value: "6" },
  { icon: LineChart, label: "Acceptance Rate", value: "~9.6%" },
];

const files = [
  {
    path: "src/loan_analysis.py",
    desc: "Main analysis pipeline — loading, cleaning, EDA, charts and printed business insights.",
  },
  {
    path: "src/eda.py",
    desc: "Reusable EDA helper functions (summaries, distributions, groupby stats) used across the project.",
  },
  {
    path: "notebooks/loan_eda.ipynb",
    desc: "Interactive Jupyter notebook for step-by-step exploratory analysis.",
  },
  {
    path: "data/Bank_Personal_Loan_Modelling.csv",
    desc: "Source dataset of 5,000 bank customers and their personal loan decisions.",
  },
  {
    path: "requirements.txt",
    desc: "pandas, numpy, matplotlib, seaborn, jupyter — install with pip install -r requirements.txt.",
  },
];

function Index() {
  return (
    <main className="min-h-screen bg-[#0b1220] text-slate-200 antialiased">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-slate-800/60">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-40"
          style={{
            background:
              "radial-gradient(60% 50% at 20% 0%, rgba(16,185,129,0.18), transparent 60%), radial-gradient(50% 40% at 85% 10%, rgba(56,189,248,0.14), transparent 60%)",
          }}
        />
        <div className="relative mx-auto max-w-5xl px-6 py-20 sm:py-28">
          <p className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1.5 text-xs font-medium tracking-wide text-emerald-300">
            <FileCode2 className="h-3.5 w-3.5" />
            Python · Pandas · NumPy · Matplotlib · Seaborn
          </p>
          <h1 className="mt-6 text-4xl font-bold leading-tight text-white sm:text-6xl">
            Personal Loan Portfolio{" "}
            <span className="bg-gradient-to-r from-emerald-400 to-sky-400 bg-clip-text text-transparent">
              Analysis
            </span>
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-relaxed text-slate-400 sm:text-lg">
            An internship data-science project exploring 5,000 bank customers to
            understand what drives personal loan acceptance — from data cleaning
            to visual storytelling and business insights.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400"
            >
              <Github className="h-4 w-4" />
              View on GitHub
            </a>
            <a
              href="#pipeline"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-5 py-2.5 text-sm font-semibold text-slate-200 transition hover:border-slate-500 hover:text-white"
            >
              Explore the analysis
            </a>
          </div>

          {/* Stats */}
          <div className="mt-14 grid grid-cols-2 gap-4 sm:grid-cols-4">
            {stats.map((s) => (
              <div
                key={s.label}
                className="rounded-xl border border-slate-800 bg-slate-900/60 p-4"
              >
                <s.icon className="h-5 w-5 text-emerald-400" />
                <p className="mt-3 text-2xl font-bold text-white">{s.value}</p>
                <p className="text-xs text-slate-400">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section id="pipeline" className="mx-auto max-w-5xl px-6 py-16 sm:py-20">
        <h2 className="text-2xl font-bold text-white sm:text-3xl">
          Analysis pipeline
        </h2>
        <p className="mt-2 text-slate-400">
          Run <code className="rounded bg-slate-800 px-1.5 py-0.5 text-emerald-300">python src/loan_analysis.py</code>{" "}
          to execute the full workflow end to end.
        </p>
        <div className="mt-10 grid gap-5 sm:grid-cols-2">
          {pipeline.map((step, i) => (
            <article
              key={step.title}
              className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 transition hover:border-emerald-500/40"
            >
              <div className="flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400">
                  <step.icon className="h-5 w-5" />
                </span>
                <h3 className="text-lg font-semibold text-white">
                  {i + 1}. {step.title}
                </h3>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-slate-400">
                {step.desc}
              </p>
            </article>
          ))}
        </div>
      </section>

      {/* Repo structure */}
      <section className="border-t border-slate-800/60 bg-slate-950/40">
        <div className="mx-auto max-w-5xl px-6 py-16 sm:py-20">
          <h2 className="text-2xl font-bold text-white sm:text-3xl">
            Repository structure
          </h2>
          <p className="mt-2 text-slate-400">
            Everything needed to reproduce the analysis lives in the repo.
          </p>
          <ul className="mt-8 space-y-3">
            {files.map((f) => (
              <li
                key={f.path}
                className="flex flex-col gap-1 rounded-lg border border-slate-800 bg-slate-900/60 p-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6"
              >
                <code className="text-sm font-medium text-emerald-300">
                  {f.path}
                </code>
                <span className="text-sm text-slate-400 sm:text-right">
                  {f.desc}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800/60">
        <div className="mx-auto flex max-w-5xl flex-col items-center gap-2 px-6 py-8 text-center text-xs text-slate-500 sm:flex-row sm:justify-between sm:text-left">
          <p>Personal Loan Portfolio Analysis — Internvia internship project</p>
          <p>Built with Python, Pandas, NumPy, Matplotlib &amp; Seaborn</p>
        </div>
      </footer>
    </main>
  );
}
