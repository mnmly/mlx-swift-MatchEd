// MatchEdDemo — a minimal SwiftUI frontend for the shared MatchEdKit.MatchEd
// pipeline.
//
// The Xcode project depends on the repo root as a local SwiftPM package
// (product "MatchEdKit"). The app contains no model / conv / weight code — it
// only drives `MatchEd` and presents its edge maps (swift-cli-gui-shared-driver).

import SwiftUI

@main
struct MatchEdDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 760, minHeight: 560)
        }
    }
}
