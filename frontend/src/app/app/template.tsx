// A template, unlike the layout, is rebuilt on every navigation, so each page plays its entrance.
export default function AppTemplate({ children }: { children: React.ReactNode }) {
  return <div className="page-enter space-y-8">{children}</div>;
}
