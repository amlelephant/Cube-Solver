import { NavBar } from "@/components/NavBar";
import { Footer } from "@/components/Footer";
import { RequireAuth } from "@/components/RequireAuth";

/**
 * Everything in this route group is signed-in-only. The landing page (`/`)
 * and the auth screens (`/auth/*`) live outside it and stay public — which is
 * also why the login page must not be moved in here, or it would guard
 * itself and redirect forever.
 */
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <RequireAuth>
      <div className="flex min-h-screen flex-col">
        <NavBar />
        <main className="flex-1">{children}</main>
        <Footer />
      </div>
    </RequireAuth>
  );
}
